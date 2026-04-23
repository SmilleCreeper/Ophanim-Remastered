# pages/model_debug.py - Comprehensive Model Analysis and Debugging

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from model import device
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr


def compute_token_loss(model, input_ids, tokenizer):
    """
    Compute per-token loss for a sequence.
    Returns list of (token_id, token_text, loss_value)
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(input_ids.unsqueeze(0))
        else:
            logits = model(input_ids.unsqueeze(0))
    
    if hasattr(logits, 'logits'):
        logits = logits.logits
    
    logits = logits[0]  # Remove batch dim
    
    # Compute loss for each token
    token_losses = []
    for i in range(len(input_ids) - 1):
        target_token = input_ids[i + 1]
        token_logits = logits[i]
        
        # Compute cross-entropy loss for this token
        loss = F.cross_entropy(token_logits.unsqueeze(0), target_token.unsqueeze(0))
        
        token_text = tokenizer.decode([target_token.item()])
        token_losses.append({
            'position': i,
            'token_id': target_token.item(),
            'token_text': token_text,
            'loss': loss.item()
        })
    
    return token_losses


def analyze_example(model, example_item, tokenizer, example_id):
    """
    Analyze a single example from dataset.
    Returns dict with example statistics.
    """
    pos_ids = example_item['positive']
    neg_ids_list = example_item['negatives']
    
    # Compute positive loss
    pos_token_losses = compute_token_loss(model, pos_ids, tokenizer)
    pos_losses = [t['loss'] for t in pos_token_losses]
    
    # Compute negative losses
    neg_losses_all = []
    for neg_ids in neg_ids_list:
        if neg_ids is None or neg_ids.numel() <= 1:
            continue
        neg_token_losses = compute_token_loss(model, neg_ids, tokenizer)
        neg_losses = [t['loss'] for t in neg_token_losses]
        neg_losses_all.extend(neg_losses)
    
    avg_neg_loss = np.mean(neg_losses_all) if neg_losses_all else 0.0
    
    return {
        'example_id': example_id,
        'num_tokens': len(pos_ids),
        'avg_loss': np.mean(pos_losses),
        'max_loss': np.max(pos_losses),
        'min_loss': np.min(pos_losses),
        'std_loss': np.std(pos_losses),
        'num_negatives': len(neg_ids_list),
        'avg_neg_loss': avg_neg_loss,
        'token_losses': pos_token_losses
    }


def analyze_all_examples(model, dataset_iterator, tokenizer, progress_bar, status_text):
    """
    Analyze all examples in dataset.
    Returns list of example statistics and token vocabulary statistics.
    """
    examples_data = []
    token_vocab_data = defaultdict(lambda: {
        'token_id': 0,
        'token_text': '',
        'total_occurrences': 0,
        'correct_predictions': 0,
        'incorrect_predictions': 0,
        'losses': []
    })
    
    total_examples = len(dataset_iterator)
    
    for idx in range(total_examples):
        status_text.text(f"Analyzing example {idx + 1}/{total_examples}...")
        progress_bar.progress((idx + 1) / total_examples)
        
        example_item = dataset_iterator[idx]
        example_stats = analyze_example(model, example_item, tokenizer, idx)
        examples_data.append(example_stats)
        
        # Collect token statistics
        for token_info in example_stats['token_losses']:
            token_id = token_info['token_id']
            token_text = token_info['token_text']
            loss = token_info['loss']
            
            token_vocab_data[token_id]['token_id'] = token_id
            token_vocab_data[token_id]['token_text'] = token_text
            token_vocab_data[token_id]['total_occurrences'] += 1
            token_vocab_data[token_id]['losses'].append(loss)
            
            # Consider "correct" if loss < 1.0
            if loss < 1.0:
                token_vocab_data[token_id]['correct_predictions'] += 1
            else:
                token_vocab_data[token_id]['incorrect_predictions'] += 1
    
    # Process token vocabulary data
    vocab_list = []
    for token_id, data in token_vocab_data.items():
        losses = data['losses']
        vocab_list.append({
            'token_id': data['token_id'],
            'token_text': data['token_text'],
            'total_occurrences': data['total_occurrences'],
            'correct_predictions': data['correct_predictions'],
            'incorrect_predictions': data['incorrect_predictions'],
            'accuracy_rate': data['correct_predictions'] / data['total_occurrences'] if data['total_occurrences'] > 0 else 0,
            'avg_loss': np.mean(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'std_loss': np.std(losses)
        })
    
    return examples_data, vocab_list


def create_categorical_columns(df):
    """
    Create categorical versions of all numeric columns (low, mid, high).
    Returns expanded dataframe with categorical columns.
    """
    expanded_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        # Skip if column has no variance or all NaN
        if df[col].isna().all() or df[col].std() == 0:
            continue
            
        # For columns with very few unique values, handle differently
        unique_vals = df[col].dropna().nunique()
        
        if unique_vals <= 2:
            # Binary or constant column - skip categorization
            continue
        elif unique_vals <= 5:
            # Few unique values - use actual values as categories
            q33 = df[col].quantile(0.33)
            q67 = df[col].quantile(0.67)
        else:
            # Normal categorization
            q33 = df[col].quantile(0.33)
            q67 = df[col].quantile(0.67)
        
        # Create binary columns for each category
        expanded_df[f'low_{col}'] = (df[col] <= q33).astype(float)
        expanded_df[f'mid_{col}'] = ((df[col] > q33) & (df[col] <= q67)).astype(float)
        expanded_df[f'high_{col}'] = (df[col] > q67).astype(float)
        
        # Replace NaN in original column with NaN in categoricals
        nan_mask = df[col].isna()
        expanded_df.loc[nan_mask, f'low_{col}'] = np.nan
        expanded_df.loc[nan_mask, f'mid_{col}'] = np.nan
        expanded_df.loc[nan_mask, f'high_{col}'] = np.nan
    
    return expanded_df


def detect_correlations(df, threshold=0.3):
    """
    Detect correlations between categorical versions of numerical columns.
    Returns list of significant correlations between low/mid/high categories.
    """
    # Create categorical columns
    expanded_df = create_categorical_columns(df)
    
    # Get all categorical columns (low_, mid_, high_ prefixed)
    categorical_cols = [col for col in expanded_df.columns if 
                       col.startswith('low_') or col.startswith('mid_') or col.startswith('high_')]
    
    correlations = []
    
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            # Skip comparing same base variable (e.g., low_avg_loss with mid_avg_loss)
            base1 = '_'.join(col1.split('_')[1:])
            base2 = '_'.join(col2.split('_')[1:])
            if base1 == base2:
                continue
            
            # Remove NaN values
            valid_mask = ~(expanded_df[col1].isna() | expanded_df[col2].isna())
            valid_count = valid_mask.sum()
            
            # Need at least 5 valid samples for meaningful correlation
            if valid_count < 5:
                continue
            
            x = expanded_df.loc[valid_mask, col1].values
            y = expanded_df.loc[valid_mask, col2].values
            
            # Check if there's any variance in the data
            if x.std() == 0 or y.std() == 0:
                continue
            
            # Check if all values are the same
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                continue
            
            try:
                pearson_corr, pearson_p = pearsonr(x, y)
                spearman_corr, spearman_p = spearmanr(x, y)
                
                # Check for invalid correlation values
                if np.isnan(pearson_corr) or np.isnan(spearman_corr):
                    continue
                
                if abs(pearson_corr) >= threshold and pearson_p < 0.05:
                    correlations.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'relationship': 'Positive' if pearson_corr > 0 else 'Negative',
                        'strength': 'Strong' if abs(pearson_corr) > 0.7 else 'Moderate' if abs(pearson_corr) > 0.5 else 'Weak',
                        'sample_size': valid_count
                    })
            except Exception as e:
                # Skip if correlation calculation fails
                continue
    
    return sorted(correlations, key=lambda x: abs(x['pearson_correlation']), reverse=True)


def get_color_for_loss(loss_value):
    """
    Get RGB color for loss value.
    Green for low loss (<0.5), yellow for medium (0.5-1.0), red for high (>1.0).
    """
    if loss_value < 0.5:
        # Green to yellow
        ratio = loss_value / 0.5
        r = int(ratio * 255)
        g = 255
        b = 0
    elif loss_value < 1.0:
        # Yellow to orange
        ratio = (loss_value - 0.5) / 0.5
        r = 255
        g = int(255 - ratio * 100)
        b = 0
    else:
        # Orange to red
        ratio = min((loss_value - 1.0) / 2.0, 1.0)
        r = 255
        g = int(155 - ratio * 155)
        b = 0
    
    return f"rgb({r}, {g}, {b})"


def display_token_by_token_analysis(example_stats):
    """
    Display token-by-token analysis with color-coded losses.
    """
    st.subheader("Token-by-Token Analysis")
    
    # Create HTML for colored text
    html_parts = ['<div style="font-family: monospace; line-height: 2.0; font-size: 14px;">']
    
    for token_info in example_stats['token_losses']:
        token_text = token_info['token_text'].replace('<', '&lt;').replace('>', '&gt;')
        loss = token_info['loss']
        color = get_color_for_loss(loss)
        
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; '
            f'border-radius: 3px;" title="Token: {token_text} | Loss: {loss:.4f}">'
            f'{token_text}</span>'
        )
    
    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
    **Legend:**
    - <span style="background-color: rgb(0, 255, 0); padding: 2px 8px;">Green</span>: Low loss (< 0.5)
    - <span style="background-color: rgb(255, 255, 0); padding: 2px 8px;">Yellow</span>: Medium loss (0.5 - 1.0)
    - <span style="background-color: rgb(255, 155, 0); padding: 2px 8px;">Orange</span>: High loss (1.0 - 2.0)
    - <span style="background-color: rgb(255, 0, 0); padding: 2px 8px;">Red</span>: Very high loss (> 2.0)
    """, unsafe_allow_html=True)
    
    # Detailed token table
    with st.expander("📊 Detailed Token Statistics"):
        token_df = pd.DataFrame(example_stats['token_losses'])
        st.dataframe(token_df, use_container_width=True)


def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for categorical columns (low/mid/high).
    """
    # Create categorical columns
    expanded_df = create_categorical_columns(df)
    
    # Get only categorical columns
    categorical_cols = [col for col in expanded_df.columns if 
                       col.startswith('low_') or col.startswith('mid_') or col.startswith('high_')]
    
    if len(categorical_cols) < 2:
        return
    
    # Compute correlation matrix, handling NaN values
    corr_matrix = expanded_df[categorical_cols].corr(method='pearson', min_periods=5)
    
    # Replace NaN with 0 for visualization (will show as white/neutral)
    corr_matrix_display = corr_matrix.fillna(0)
    
    # Create text annotations - show actual values or "N/A" for NaN
    text_matrix = []
    for i in range(len(corr_matrix)):
        row = []
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if pd.isna(val):
                row.append("N/A")
            else:
                row.append(f"{val:.2f}")
        text_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix_display.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 7},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Heatmap (Categorical Levels)",
        xaxis_title="",
        yaxis_title="",
        height=800,
        xaxis={'tickangle': -45, 'tickfont': {'size': 9}},
        yaxis={'tickfont': {'size': 9}}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_loss_distribution(examples_df):
    """
    Plot distribution of average losses across examples.
    """
    fig = px.histogram(
        examples_df,
        x='avg_loss',
        nbins=50,
        title='Distribution of Average Loss Across Examples',
        labels={'avg_loss': 'Average Loss', 'count': 'Number of Examples'}
    )
    
    fig.add_vline(x=examples_df['avg_loss'].median(), line_dash="dash", 
                  line_color="red", annotation_text="Median")
    
    st.plotly_chart(fig, use_container_width=True)


def plot_token_accuracy_distribution(vocab_df):
    """
    Plot distribution of token accuracy rates.
    """
    fig = px.histogram(
        vocab_df,
        x='accuracy_rate',
        nbins=50,
        title='Distribution of Token Accuracy Rates',
        labels={'accuracy_rate': 'Accuracy Rate', 'count': 'Number of Tokens'}
    )
    
    fig.add_vline(x=vocab_df['accuracy_rate'].median(), line_dash="dash",
                  line_color="red", annotation_text="Median")
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.header("🔍 Model Debug & Analysis")
    st.markdown("Comprehensive analysis of model performance on the training dataset.")
    
    # Check prerequisites
    if st.session_state.model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    if st.session_state.dataset_iterator is None:
        st.warning("⚠️ No dataset loaded. Please load a dataset first.")
        return
    
    # Analysis button
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    if 'examples_data' not in st.session_state:
        st.session_state.examples_data = None
    
    if 'vocab_data' not in st.session_state:
        st.session_state.vocab_data = None
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            with st.spinner("Analyzing all examples..."):
                examples_data, vocab_data = analyze_all_examples(
                    st.session_state.model,
                    st.session_state.dataset_iterator,
                    st.session_state.tokenizer,
                    progress_bar,
                    status_text
                )
            
            elapsed_time = time.time() - start_time
            
            st.session_state.examples_data = examples_data
            st.session_state.vocab_data = vocab_data
            st.session_state.analysis_complete = True
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Analysis complete! Processed {len(examples_data)} examples in {elapsed_time:.2f}s")
    
    with col2:
        if st.session_state.analysis_complete:
            if st.button("🔄 Clear Analysis", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.examples_data = None
                st.session_state.vocab_data = None
                st.rerun()
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.examples_data is not None:
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Examples Overview", "📖 Vocabulary Analysis", "🔗 Correlations", "🔎 Detailed Example"])
        
        with tab1:
            st.subheader("Examples Performance Analysis")
            
            # Convert to DataFrame
            examples_df = pd.DataFrame(st.session_state.examples_data)
            examples_df = examples_df.drop('token_losses', axis=1)  # Remove nested data for display
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Examples", len(examples_df))
                st.metric("Avg Loss (Mean)", f"{examples_df['avg_loss'].mean():.4f}")
            with col2:
                st.metric("Avg Tokens/Example", f"{examples_df['num_tokens'].mean():.1f}")
                st.metric("Avg Loss (Median)", f"{examples_df['avg_loss'].median():.4f}")
            with col3:
                st.metric("Best Example Loss", f"{examples_df['avg_loss'].min():.4f}")
                st.metric("Worst Example Loss", f"{examples_df['avg_loss'].max():.4f}")
            with col4:
                good_examples = (examples_df['avg_loss'] < 1.0).sum()
                st.metric("Good Examples (<1.0)", good_examples)
                st.metric("% Good", f"{100 * good_examples / len(examples_df):.1f}%")
            
            # Distribution plot
            plot_loss_distribution(examples_df)
            
            # Sortable table
            st.subheader("📋 Examples Spreadsheet")
            st.dataframe(
                examples_df.style.background_gradient(subset=['avg_loss'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = examples_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Examples CSV",
                data=csv,
                file_name="examples_analysis.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("Token Vocabulary Analysis")
            
            # Convert to DataFrame
            vocab_df = pd.DataFrame(st.session_state.vocab_data)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Unique Tokens", len(vocab_df))
                st.metric("Avg Occurrences", f"{vocab_df['total_occurrences'].mean():.1f}")
            with col2:
                st.metric("Avg Accuracy", f"{vocab_df['accuracy_rate'].mean():.2%}")
                st.metric("Median Accuracy", f"{vocab_df['accuracy_rate'].median():.2%}")
            with col3:
                high_accuracy = (vocab_df['accuracy_rate'] > 0.8).sum()
                st.metric("High Accuracy Tokens (>80%)", high_accuracy)
                st.metric("% High Accuracy", f"{100 * high_accuracy / len(vocab_df):.1f}%")
            with col4:
                st.metric("Best Token Accuracy", f"{vocab_df['accuracy_rate'].max():.2%}")
                st.metric("Worst Token Accuracy", f"{vocab_df['accuracy_rate'].min():.2%}")
            
            # Distribution plot
            plot_token_accuracy_distribution(vocab_df)
            
            # Sortable table
            st.subheader("📋 Vocabulary Spreadsheet")
            st.dataframe(
                vocab_df.style.background_gradient(subset=['accuracy_rate'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = vocab_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Vocabulary CSV",
                data=csv,
                file_name="vocabulary_analysis.csv",
                mime="text/csv"
            )
            
            # Top/Bottom performers
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Top 10 Best Performing Tokens")
                top_tokens = vocab_df.nlargest(10, 'accuracy_rate')[['token_text', 'accuracy_rate', 'avg_loss', 'total_occurrences']]
                st.dataframe(top_tokens, use_container_width=True)
            
            with col2:
                st.subheader("⚠️ Top 10 Worst Performing Tokens")
                worst_tokens = vocab_df.nsmallest(10, 'accuracy_rate')[['token_text', 'accuracy_rate', 'avg_loss', 'total_occurrences']]
                st.dataframe(worst_tokens, use_container_width=True)
        
        with tab3:
            st.subheader("Correlation Analysis")
            st.markdown("Detecting relationships between different metrics.")
            
            # Detect correlations
            examples_df_numeric = pd.DataFrame(st.session_state.examples_data).drop('token_losses', axis=1)
            correlations = detect_correlations(examples_df_numeric, threshold=0.3)
            
            if correlations:
                st.success(f"Found {len(correlations)} significant correlations between categorical levels!")
                
                # Display correlation table
                corr_df = pd.DataFrame(correlations)
                
                st.dataframe(
                    corr_df.style.background_gradient(
                        subset=['pearson_correlation'], cmap='RdYlGn', vmin=-1, vmax=1
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Heatmap
                plot_correlation_heatmap(examples_df_numeric)
                
                # Key insights
                st.subheader("🔑 Key Insights")
                for i, corr in enumerate(correlations[:10]):
                    relationship_type = "positively" if corr['relationship'] == 'Positive' else "negatively"
                    
                    # Parse variable names for better readability
                    var1_parts = corr['variable_1'].split('_')
                    var2_parts = corr['variable_2'].split('_')
                    
                    level1 = var1_parts[0].upper()
                    metric1 = '_'.join(var1_parts[1:])
                    level2 = var2_parts[0].upper()
                    metric2 = '_'.join(var2_parts[1:])
                    
                    st.info(
                        f"**{level1} {metric1}** is {relationship_type} correlated with "
                        f"**{level2} {metric2}** "
                        f"({corr['strength']}, r={corr['pearson_correlation']:.3f}, p={corr['pearson_p_value']:.4f})"
                    )
            else:
                st.info("No significant correlations found (threshold: 0.3)")
        
        with tab4:
            st.subheader("Detailed Example Inspection")
            
            # Dropdown to select example
            example_ids = list(range(len(st.session_state.examples_data)))
            selected_id = st.selectbox(
                "Select Example ID",
                options=example_ids,
                format_func=lambda x: f"Example {x} (Avg Loss: {st.session_state.examples_data[x]['avg_loss']:.4f})"
            )
            
            if selected_id is not None:
                example_stats = st.session_state.examples_data[selected_id]
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Example ID", selected_id)
                    st.metric("Num Tokens", example_stats['num_tokens'])
                with col2:
                    st.metric("Average Loss", f"{example_stats['avg_loss']:.4f}")
                    st.metric("Std Dev Loss", f"{example_stats['std_loss']:.4f}")
                with col3:
                    st.metric("Max Loss", f"{example_stats['max_loss']:.4f}")
                    st.metric("Min Loss", f"{example_stats['min_loss']:.4f}")
                with col4:
                    st.metric("Num Negatives", example_stats['num_negatives'])
                    st.metric("Avg Neg Loss", f"{example_stats['avg_neg_loss']:.4f}")
                
                # Token-by-token visualization
                display_token_by_token_analysis(example_stats)
                
                # Loss progression chart
                st.subheader("📈 Loss Progression")
                token_losses_list = [t['loss'] for t in example_stats['token_losses']]
                positions = list(range(len(token_losses_list)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=positions,
                    y=token_losses_list,
                    mode='lines+markers',
                    name='Token Loss',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                             annotation_text="Threshold (1.0)")
                fig.add_hline(y=example_stats['avg_loss'], line_dash="dot", 
                             line_color="green", annotation_text="Average")
                
                fig.update_layout(
                    title="Per-Token Loss Progression",
                    xaxis_title="Token Position",
                    yaxis_title="Loss",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()