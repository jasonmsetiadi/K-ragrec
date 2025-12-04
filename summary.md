### Initialize Models

model: 'meta-llama/Llama-2-7b-hf' (`model/graph_llm.py`)
- tokenizer: AutoTokenizer from transformers
- model: AutoModelForCausalLM from transformers
- graph_encoder: gnn.py self-code using torch and torch_geometric (`from torch_geometric.nn import GCNConv, **TransformerConv**, GATConv,  SAGEConv`)
- projector: torch.nn.Sequential

retrieval_model: 'sentence-transformers/all-roberta-large-v1' (`processing_kg/lm_modeling.py`)
- tokenizer: AutoTokenizer from transformers
- model: AutoModel from transformers
- G: torch_geometric.Data object (`index_KG.py`)
- G1: torch_geometric.Data object (`index_KG.py`)
- Graph: networkx.Graph

### Retrieval 
input data:
- input: (str) user watch history (10 movies)
- sequence_ids: (list[int]) freebase id of user watch history

output data: 
- subgraph: torch_geometric.Data object (combined 1st and 2nd order subgraphs)

process: (`retrieve.py`)
1. popularity selective retrieval policy  
```
retrieve_movies_list = retrieval_model.whether_retrieval(args.adaptive_ratio*sequence_id, 5)
```
- i dont get why need to replicate sequence_id by adaptive ratio times?
- `sort_item_list:71` = from 10 movies in user watch history, only retain the K (5) least popular items (least rated by users)

2. retrieve 1st and 2nd order subgraph for the remaining items
```
retrieval_model.retrieval_topk(input, retrieve_movies_list, args.sub_graph_numbers, args.reranking_numbers)
```
- N items retrieves N * 2 subgraphs (in our case 5 items retrieves 10 subgraphs)
- `retrieval_topk:226` = for each movie in the filtered set, encode to text embedding (using retrieval_model) and compute cosine similarity on the first and second order graphs to get their top K nodes (default = 3, so total 6 nodes / movie)
- `re_ranking:195` = filter the retrieved first and second order nodes by obtaining top K (5) nodes closest to the user watch history (input) in the embedding space using cosine similarity (end with K first order nodes and K second order nodes)
- `get_first_order_subgraph:107` = given the filtered first and second order nodes, construct subgraph for each node (end with K first order subgraphs and K second order subgraphs)


### Inference
input data:
- input: (str) user watch history (10 movies)
- questions: (str) 20 candidate movies (A-T)
- graph: list of torch_geometric.Data object (combined 1st and 2nd order subgraphs)

output data: 

total inference 21.5 minutes / sample (CPU) but 2 minutes / sample (MPS)

process: (`graph_llm.py`)
1. Encode text prompt (instructions + history + candidates)
- tokenize text prompt using `model.tokenizer`
- append end of sentence token to the text prompt tokens `questions.input_ids[i] + eos_user_tokens.input_ids`
- embed the tokens using `model.model.get_input_embeddings()`

2. Encode retrieved graphs 

for each subgraph:
- obtain node embeddings through GNN forward pass given graph data (node features, edges, edge features)
```
n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
```
- pool node embeddings into graph embeddings (compute mean of node embeddings), has dimension 1024
```
g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
```
- projects GNN dimension (1024) to LLM dimension (4096) using trained 2 layer MLP
```        
projected_graph_embeds_list = [self.projector(g_embeds) for g_embeds in graph_embeds_list]
```
- mean pool them into single vector (10, 4096) -> (1, 4096)
```           
sample_graph_embeds = sample_graph_embeds.mean(dim=0, keepdim=True)
```

3. Combine Embeddings
- the final embedding for each sample consist of `[BOS] + [GRAPH_EMBED] + [TEXT_EMBEDS]`
- given we generate in batches of samples, we need to make them uniform through padding since each sample have different token lengths
```
# pad inputs_embeds
max_length = max([x.shape[0] for x in batch_inputs_embeds])
for i in range(batch_size):
    pad_length = max_length-batch_inputs_embeds[i].shape[0]
    batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
    batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
```

4 Generation
```
generation_output = self.model.generate(
    inputs_embeds=inputs_embeds,
    max_new_tokens=self.max_new_tokens, 
    attention_mask=attention_mask,
    return_dict_in_generate=True, 
    output_scores=True,
    use_cache=True 
)
```
- `generation_output.sequences` contains the generated token (only first token is used)
- `generation_output.scores[0].softmax(dim=-1)` turns scores of all tokens in vocab size (32000) into probabilities
- `torch.tensor(scores[:, option_indices], dtype=torch.float32).softmax(dim=-1)` filters the 20 tokens (A-T) and turn to probabilities again
- `specific_logits.argsort(dim=-1, descending=True)` finally we output in descending order

### Training
1-3 same steps as Inference

4. Forward
- initialize
```
params = [p for _, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
[{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
betas=(0.9, 0.95)
)
optimizer.zero_grad()
```
- forward pass
```
outputs = self.model(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    return_dict=True,
    labels=label_input_ids,
    # do_sample=True
)
```
- backward pass `loss.backward()`
- parameter update `optimizer.step()`

### Metrics Computation
input data:
- ground_truth: (int) label / recommended movie (0-19)
- output: list of probabilities (20)

outputs:
- recall@k, k=1,3,5,10
```
recall_1,recall_3,recall_5,recall_10 = recall_at_k([ground_truth[ind]], output[ind], 10)
```
