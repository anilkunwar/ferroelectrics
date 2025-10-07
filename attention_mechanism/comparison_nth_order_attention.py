import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch  # Assuming PyTorch for tensor operations; user can install if needed
import itertools

st.set_page_config(page_title="Higher-Order Attention Tutorial", layout="wide")

# Sidebar for parameters
st.sidebar.title("Parameters")
sequence_length = st.sidebar.slider("Sequence Length (n)", min_value=3, max_value=10, value=5, help="Number of tokens in the sequence. Keep small to avoid computation explosion.")
attention_order = st.sidebar.selectbox("Attention Order", options=["Pairwise (2)", "Triplet (3)", "Quadruple (4)", "Infinity (Approximated)"], help="Select the order of attention to compare.")
compute_button = st.sidebar.button("Compute and Compare")

# Main content
st.title("Higher-Order Attention Tutorial for Research in Ferroelectricity")
st.markdown("""
This Streamlit app provides a simple tutorial to parametrically compare pairwise, triplet, quadruple, and 'infinity' attention mechanisms, inspired by the application to Quantitative Named Entity Recognition (NER) in ferroelectricity literature.

### Background
In standard NER for materials science (e.g., ferroelectricity papers), we identify entities like materials (PbTiO₃), properties (coercive field), values (350 kV/cm), etc. Higher-order attention helps capture complex relationships beyond pairs, such as tuples involving material, property, value, and conditions.

- **Pairwise Attention (Order 2)**: Standard self-attention, computes interactions between every pair of tokens (O(n²)).
- **Triplet Attention (Order 3)**: Computes interactions for every triplet of tokens (O(n³)).
- **Quadruple Attention (Order 4)**: For quadruplets (O(n⁴)).
- **Infinity Attention**: Approximated here as multi-layer pairwise attention (e.g., 3 layers) to simulate compositional higher-order effects without exponential cost.

We'll use a toy example with random embeddings for a small sequence. For visualization:
- Order 2: Heatmap of attention matrix.
- Higher orders: Sliced heatmaps (fixing one dimension) or aggregated scores, as full tensors are hard to visualize.
- Complexity: We'll show computational cost and feasibility.

Sample Sentence (tokenized for demo): "The gradient energy coefficient μ for PbTiO₃ was 1.5 × 10⁻¹⁰ C⁻² m⁴ N."
For simplicity, we'll simulate with n random tokens.
""")

if compute_button:
    st.header("Computation Results")
    
    # Generate random embeddings (toy example)
    d_model = 64  # Embedding dimension
    tokens = [f"Token {i+1}" for i in range(sequence_length)]
    embeddings = torch.randn(sequence_length, d_model)
    
    # Function to compute softmax-normalized scores
    def softmax(x, dim=-1):
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
        return e_x / e_x.sum(dim=dim, keepdim=True)
    
    order = {
        "Pairwise (2)": 2,
        "Triplet (3)": 3,
        "Quadruple (4)": 4,
        "Infinity (Approximated)": "inf"
    }[attention_order]
    
    if order == "inf":
        st.subheader("Infinity Attention (Approximated via Multi-Layer Pairwise)")
        st.markdown("For 'infinity', we approximate higher-order effects using 3 layers of standard pairwise attention (compositional buildup).")
        
        # Simulate multi-layer attention
        def pairwise_attention(X):
            Q = X @ torch.randn(d_model, d_model)
            K = X @ torch.randn(d_model, d_model)
            scores = softmax(Q @ K.T / np.sqrt(d_model))
            return scores
        
        attn_layer1 = pairwise_attention(embeddings)
        attn_layer2 = pairwise_attention(attn_layer1 @ embeddings)  # Attention over attention
        attn_layer3 = pairwise_attention(attn_layer2 @ embeddings)
        
        # Visualize final attention matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(attn_layer3.numpy(), annot=True if sequence_length <= 5 else False, cmap="YlGnBu", ax=ax)
        ax.set_title("Approximated Infinity Attention Matrix")
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens, rotation=0)
        st.pyplot(fig)
        
        st.markdown(f"Computational Complexity: Approximately O(3 * n²) = O({3 * sequence_length**2})")
    else:
        st.subheader(f"{attention_order} Attention")
        st.markdown(f"Computing attention over all {order}-tuples. Complexity: O(n^{order}) = O({sequence_length}^{order}).")
        
        if sequence_length ** order > 100000:  # Arbitrary threshold to avoid long computation
            st.warning(f"Computation too expensive (>{sequence_length**order} operations). Showing conceptual demo only.")
            st.markdown("In practice, use approximations like factored or sparse attention for higher orders.")
        else:
            # Compute higher-order attention scores (toy: average dot-products over tuples)
            st.markdown("Toy implementation: For each k-tuple, score = average pairwise dot-product within the tuple.")
            scores = {}
            for tuple_idx in itertools.product(range(sequence_length), repeat=order):
                tuple_embs = embeddings[list(tuple_idx)]
                pair_scores = [torch.dot(tuple_embs[i], tuple_embs[j]) for i in range(order) for j in range(i+1, order)]
                avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0
                scores[tuple_idx] = float(softmax(torch.tensor([avg_score]))[0])  # Normalize for demo
            
            # Visualize: For order=2, full matrix; for higher, show top-5 scores and a sliced view
            if order == 2:
                attn_matrix = np.zeros((sequence_length, sequence_length))
                for (i, j), score in scores.items():
                    attn_matrix[i, j] = score
                    attn_matrix[j, i] = score  # Symmetric for demo
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(attn_matrix, annot=True if sequence_length <= 5 else False, cmap="YlGnBu", ax=ax)
                ax.set_title("Pairwise Attention Matrix")
                ax.set_xticklabels(tokens, rotation=45, ha="right")
                ax.set_yticklabels(tokens, rotation=0)
                st.pyplot(fig)
            else:
                # Show top-5 tuples by score
                top_tuples = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                st.subheader("Top 5 High-Scoring Tuples")
                for tuple_idx, score in top_tuples:
                    tuple_labels = [tokens[i] for i in tuple_idx]
                    st.markdown(f"Tuple: {tuple_labels} | Score: {score:.4f}")
                
                # Sliced visualization: Fix first index to 0, show remaining as matrix (for order=3/4)
                if order == 3:
                    slice_matrix = np.zeros((sequence_length, sequence_length))
                    for i in range(sequence_length):
                        for j in range(sequence_length):
                            slice_matrix[i, j] = scores.get((0, i, j), 0)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(slice_matrix, annot=True if sequence_length <= 5 else False, cmap="YlGnBu", ax=ax)
                    ax.set_title("Triplet Attention Slice (Fixed Token 1)")
                    ax.set_xticklabels(tokens, rotation=45, ha="right")
                    ax.set_yticklabels(tokens, rotation=0)
                    st.pyplot(fig)
                elif order == 4:
                    st.markdown("For quadruple, visualization is challenging. Showing aggregated scores per token.")
                    agg_scores = np.zeros(sequence_length)
                    for tuple_idx, score in scores.items():
                        agg_scores[list(tuple_idx)] += score
                    agg_scores /= (sequence_length ** (order - 1))  # Normalize
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(tokens, agg_scores)
                    ax.set_title("Aggregated Quadruple Attention Scores per Token")
                    ax.set_xticklabels(tokens, rotation=45, ha="right")
                    st.pyplot(fig)
            
            st.markdown(f"Total Interactions Computed: {sequence_length ** order}")

st.markdown("""
### Key Insights
- As order increases, we capture more complex relationships (e.g., material-property-value-unit tuples in ferroelectricity texts).
- But complexity explodes: For n=10, order=4 is 10,000 operations; for real sequences (n=512), it's infeasible without approximations.
- In practice, use techniques like multi-hop attention or graph-based methods for higher-order reasoning.
- Application: Improves inference in scientific literature, e.g., extracting quantitative properties from papers on PbTiO₃ or other ferroelectrics.

For full implementation, extend this to real models like SciBERT with entity pooling and relation scoring as in the scratch idea.
""")
