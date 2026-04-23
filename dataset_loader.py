import torch
import streamlit as st

class FastDatasetIterator:
    def __init__(self, dataset, tokenizer, breakdown_delimiter="", enable_breakdown=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.breakdown_delimiter = breakdown_delimiter
        self.enable_breakdown = enable_breakdown
        self.processed_data = []
        self._preprocess_all()
    
    def _find_breakdown_index(self, text, tokens):
        """
        Find the token index after the LAST occurrence of breakdown_delimiter.
        Returns the index where learning should start, or 0 if delimiter not found.
        """
        if not self.enable_breakdown or not self.breakdown_delimiter:
            return 0
        
        # Find last occurrence of delimiter in text
        last_pos = text.rfind(self.breakdown_delimiter)
        
        if last_pos == -1:
            # Delimiter not found, learn from beginning
            return 0
        
        # Calculate position after the delimiter
        breakdown_text_pos = last_pos + len(self.breakdown_delimiter)
        
        # Tokenize up to breakdown point to find token index
        text_before = text[:breakdown_text_pos]
        tokens_before = self.tokenizer(text_before, truncation=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        # The breakdown index is where tokens_before ends
        breakdown_token_idx = tokens_before.size(0)
        
        return breakdown_token_idx
    
    def _preprocess_all(self):
        st.info("Pre-processing dataset...")
        progress_bar = st.progress(0)
        
        if self.enable_breakdown and self.breakdown_delimiter:
            st.info(f"🎯 Breakdown enabled: Learning only after '{self.breakdown_delimiter}'")
        
        # Statistics for breakdown analysis
        total_pos_tokens = 0
        total_pos_learnable_tokens = 0
        total_neg_tokens = 0
        total_neg_learnable_tokens = 0
        samples_with_breakdown = 0
        samples_without_breakdown = 0
        
        for idx, item in enumerate(self.dataset):
            # Tokenize positive example
            pos_text = item['positive_example']
            pos_tokens = self.tokenizer(pos_text, truncation=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
            pos_tokens = pos_tokens.cpu()  # Ensure CPU storage
            
            # Find breakdown index for positive example
            pos_breakdown_idx = self._find_breakdown_index(pos_text, pos_tokens)
            
            # Update statistics
            pos_token_count = pos_tokens.size(0)
            total_pos_tokens += pos_token_count
            pos_learnable = pos_token_count - pos_breakdown_idx
            total_pos_learnable_tokens += pos_learnable
            
            if pos_breakdown_idx > 0:
                samples_with_breakdown += 1
            else:
                samples_without_breakdown += 1
            
            # Process negative examples
            neg_examples = []
            neg_breakdown_indices = []
            i = 1
            while f'negative_example_{i}' in item:
                neg_text = item[f'negative_example_{i}']
                neg_tokens = self.tokenizer(neg_text, truncation=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
                neg_tokens = neg_tokens.cpu()  # Ensure CPU storage
                
                # Find breakdown index for negative example
                neg_breakdown_idx = self._find_breakdown_index(neg_text, neg_tokens)
                
                # Update statistics
                neg_token_count = neg_tokens.size(0)
                total_neg_tokens += neg_token_count
                neg_learnable = neg_token_count - neg_breakdown_idx
                total_neg_learnable_tokens += neg_learnable
                
                neg_examples.append(neg_tokens)
                neg_breakdown_indices.append(neg_breakdown_idx)
                i += 1
            
            self.processed_data.append({
                'positive': pos_tokens,
                'positive_breakdown_idx': pos_breakdown_idx,
                'negatives': neg_examples,
                'negative_breakdown_indices': neg_breakdown_indices
            })
            
            if idx % 10 == 0:
                progress_bar.progress((idx + 1) / len(self.dataset))
        
        progress_bar.empty()
        
        if self.enable_breakdown and self.breakdown_delimiter:
            st.success(f"Pre-processed {len(self.processed_data)} examples with breakdown analysis (stored in CPU RAM)")
            
            # Display breakdown statistics
            st.markdown("### 📊 Breakdown Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples with Breakdown", samples_with_breakdown)
            with col2:
                st.metric("Samples without Breakdown", samples_without_breakdown)
            with col3:
                coverage_pct = (samples_with_breakdown / len(self.processed_data) * 100) if len(self.processed_data) > 0 else 0
                st.metric("Breakdown Coverage", f"{coverage_pct:.1f}%")
            
            st.markdown("#### Positive Examples:")
            pos_col1, pos_col2, pos_col3 = st.columns(3)
            with pos_col1:
                st.metric("Total Tokens", f"{total_pos_tokens:,}")
            with pos_col2:
                st.metric("Learnable Tokens", f"{total_pos_learnable_tokens:,}")
            with pos_col3:
                pos_learnable_pct = (total_pos_learnable_tokens / total_pos_tokens * 100) if total_pos_tokens > 0 else 0
                st.metric("Learnable %", f"{pos_learnable_pct:.1f}%")
            
            if total_neg_tokens > 0:
                st.markdown("#### Negative Examples:")
                neg_col1, neg_col2, neg_col3 = st.columns(3)
                with neg_col1:
                    st.metric("Total Tokens", f"{total_neg_tokens:,}")
                with neg_col2:
                    st.metric("Learnable Tokens", f"{total_neg_learnable_tokens:,}")
                with neg_col3:
                    neg_learnable_pct = (total_neg_learnable_tokens / total_neg_tokens * 100) if total_neg_tokens > 0 else 0
                    st.metric("Learnable %", f"{neg_learnable_pct:.1f}%")
            
            # Show average breakdown point
            avg_pos_breakdown = pos_breakdown_idx if samples_with_breakdown > 0 else 0
            avg_pos_total = total_pos_tokens / len(self.processed_data) if len(self.processed_data) > 0 else 0
            
            st.markdown("#### Average Token Distribution:")
            breakdown_col1, breakdown_col2 = st.columns(2)
            with breakdown_col1:
                st.info(f"📖 **Context (ignored)**: {avg_pos_breakdown:.1f} tokens on average")
            with breakdown_col2:
                avg_learnable = avg_pos_total - avg_pos_breakdown
                st.success(f"🎯 **Response (learned)**: {avg_learnable:.1f} tokens on average")
            
            # Warning if breakdown coverage is low
            if coverage_pct < 50:
                st.warning(f"⚠️ Warning: Only {coverage_pct:.1f}% of samples contain the breakdown delimiter. "
                          f"The remaining {100-coverage_pct:.1f}% will learn from the entire sequence.")
            
        else:
            st.success(f"Pre-processed {len(self.processed_data)} examples (stored in CPU RAM)")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        """Return pre-processed item from CPU RAM - will be moved to GPU in loss functions"""
        return self.processed_data[idx]