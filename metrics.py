


def compute_fidelity(test_dataloader, gnnNets):
    """
    Compute fidelity metrics following PyG's GraphFramEx formulation.

    For model explanations (PGIB explains its own predictions):
        fid+ = 1 - (1/N) * sum( 1(y_complement == y_orig) )
        fid- = 1 - (1/N) * sum( 1(y_subgraph == y_orig) )

    The explanation subgraph is defined by the model's active_node_index
    (nodes selected by the Gumbel-softmax assignment matrix).
    Masking is done by zero-filling node features (same as my_mcts.py).

    High fid+ = removing explanation changes predictions (explanation is necessary/faithful).
    Low fid-  = keeping only explanation preserves predictions (explanation is sufficient).
    """
    complement_matches = []
    subgraph_matches = []

    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(model_args.device)

            # 1. Original prediction on full graph
            logits_orig, _, active_node_index, _, _, _, _, _ = gnnNets(batch)
            _, y_pred_orig = torch.max(logits_orig, dim=-1)

            # 2. Build a global node mask from active_node_index
            num_nodes = batch.x.size(0)
            node_mask = torch.zeros(num_nodes, device=model_args.device)

            node_offsets = [0]
            for i in range(batch.batch[-1].item() + 1):
                node_offsets.append(
                    node_offsets[-1] + (batch.batch == i).sum().item()
                )

            for i, nodes in enumerate(active_node_index):
                if isinstance(nodes, int):
                    nodes = [nodes]
                elif not isinstance(nodes, list):
                    nodes = (
                        [nodes] if not hasattr(nodes, '__len__') else list(nodes)
                    )
                for n in nodes:
                    node_mask[node_offsets[i] + n] = 1.0

            # 3. Subgraph-only prediction (for fid-)
            x_sub = batch.x * node_mask.unsqueeze(1)
            data_sub = Batch(
                x=x_sub, edge_index=batch.edge_index, batch=batch.batch
            )
            logits_sub, _, _, _, _, _, _, _ = gnnNets(data_sub)
            _, y_pred_sub = torch.max(logits_sub, dim=-1)

            # 4. Complement-only prediction (for fid+)
            x_comp = batch.x * (1.0 - node_mask).unsqueeze(1)
            data_comp = Batch(
                x=x_comp, edge_index=batch.edge_index, batch=batch.batch
            )
            logits_comp, _, _, _, _, _, _, _ = gnnNets(data_comp)
            _, y_pred_comp = torch.max(logits_comp, dim=-1)

            complement_matches.append(
                (y_pred_comp == y_pred_orig).float().cpu().numpy()
            )
            subgraph_matches.append(
                (y_pred_sub == y_pred_orig).float().cpu().numpy()
            )

    pos_fidelity = 1.0 - np.concatenate(complement_matches).mean()
    neg_fidelity = 1.0 - np.concatenate(subgraph_matches).mean()

    # Characterization score (weighted harmonic mean, GraphFramEx Eq.)
    if pos_fidelity > 0 and neg_fidelity < 1:
        charact = 1.0 / (0.5 / pos_fidelity + 0.5 / (1.0 - neg_fidelity))
    else:
        charact = 0.0

    return pos_fidelity, neg_fidelity, charact