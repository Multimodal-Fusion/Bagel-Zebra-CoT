#!/usr/bin/env python3
"""
Zebra-CoT Dataset Visualization App
Interactive Streamlit app for browsing and visualizing the Zebra-CoT dataset
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Zebra-CoT Dataset Viewer",
    page_icon="🦓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .reasoning-text {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .answer-text {
        background-color: #e8f4ea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .question-text {
        background-color: #ffe8e8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset_categories():
    """Load available dataset categories"""
    dataset_dir = Path("../datasets/Zebra-CoT")
    categories = {}
    
    for folder in dataset_dir.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            parquet_files = list(folder.glob("*.parquet"))
            if parquet_files:
                categories[folder.name] = parquet_files
    
    return categories

@st.cache_data
def load_parquet_file(file_path):
    """Load a parquet file"""
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def display_image(image_data, caption=""):
    """Display an image from bytes data"""
    if image_data and isinstance(image_data, dict) and 'bytes' in image_data:
        try:
            img = Image.open(io.BytesIO(image_data['bytes']))
            st.image(img, caption=caption, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    elif image_data:
        st.warning("Image data format not recognized")

def truncate_text(text, max_chars=500):
    """Truncate text with ellipsis if too long"""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def main():
    st.title("🦓 Zebra-CoT Dataset Viewer")
    st.markdown("### Explore the Multimodal Chain-of-Thought Reasoning Dataset")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Dataset Navigation")
        
        # Load categories
        categories = load_dataset_categories()
        
        if not categories:
            st.error("No dataset found. Please ensure the dataset is downloaded in ../datasets/Zebra-CoT")
            return
        
        # Category selection
        selected_category = st.selectbox(
            "Select Category",
            options=list(categories.keys()),
            help="Choose a dataset category to explore"
        )
        
        # File selection within category
        if selected_category:
            files = categories[selected_category]
            file_names = [f.name for f in files]
            selected_file = st.selectbox(
                "Select File",
                options=file_names,
                help=f"Found {len(files)} file(s) in this category"
            )
            
            # Load the selected file
            if selected_file:
                file_path = next(f for f in files if f.name == selected_file)
                df = load_parquet_file(file_path)
                
                if df is not None:
                    st.success(f"Loaded {len(df)} samples")
                    
                    # Sample selection
                    st.subheader("Sample Selection")
                    sample_idx = st.number_input(
                        "Sample Index",
                        min_value=0,
                        max_value=len(df)-1,
                        value=0,
                        step=1,
                        help=f"Navigate through {len(df)} samples"
                    )
                    
                    # Random sample button
                    if st.button("🎲 Random Sample"):
                        sample_idx = np.random.randint(0, len(df))
                        st.rerun()
    
    # Main content area
    if 'df' in locals() and df is not None:
        st.header(f"Category: {selected_category}")
        st.subheader(f"Sample {sample_idx + 1} of {len(df)}")
        
        # Get the sample
        sample = df.iloc[sample_idx]
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Question", "🤔 Reasoning", "✅ Answer", "📊 Metadata"])
        
        with tab1:
            st.markdown("### Question")
            if 'Question' in sample:
                st.markdown(f'<div class="question-text">{sample["Question"]}</div>', unsafe_allow_html=True)
            
            # Display problem images
            st.markdown("### Problem Images")
            cols = st.columns(4)
            for i, col in enumerate(cols, 1):
                img_key = f'problem_image_{i}'
                if img_key in sample and sample[img_key] is not None:
                    with col:
                        display_image(sample[img_key], f"Problem Image {i}")
        
        with tab2:
            st.markdown("### Chain-of-Thought Reasoning")
            if 'Text Reasoning Trace' in sample:
                # Show full reasoning or truncated with expand option
                reasoning_text = sample['Text Reasoning Trace']
                
                show_full = st.checkbox("Show full reasoning trace", value=False)
                
                if show_full:
                    st.markdown(f'<div class="reasoning-text">{reasoning_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="reasoning-text">{truncate_text(reasoning_text, 1000)}</div>', 
                              unsafe_allow_html=True)
            
            # Display reasoning images
            st.markdown("### Reasoning Images")
            cols = st.columns(4)
            for i, col in enumerate(cols, 1):
                img_key = f'reasoning_image_{i}'
                if img_key in sample and sample[img_key] is not None:
                    with col:
                        display_image(sample[img_key], f"Reasoning Image {i}")
        
        with tab3:
            st.markdown("### Final Answer")
            if 'Final Answer' in sample:
                st.markdown(f'<div class="answer-text">{sample["Final Answer"]}</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### Sample Metadata")
            
            # Display basic statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Question Length", len(sample.get('Question', '')))
            with col2:
                st.metric("Reasoning Length", len(sample.get('Text Reasoning Trace', '')))
            with col3:
                st.metric("Answer Length", len(sample.get('Final Answer', '')))
            
            # Count images
            problem_images = sum(1 for i in range(1, 10) 
                               if f'problem_image_{i}' in sample and sample[f'problem_image_{i}'] is not None)
            reasoning_images = sum(1 for i in range(1, 10) 
                                 if f'reasoning_image_{i}' in sample and sample[f'reasoning_image_{i}'] is not None)
            
            st.markdown(f"**Problem Images:** {problem_images}")
            st.markdown(f"**Reasoning Images:** {reasoning_images}")
            
            # Show all column names
            with st.expander("View all columns"):
                st.write(df.columns.tolist())
            
            # Show raw data
            with st.expander("View raw sample data"):
                # Convert sample to dict, excluding image bytes for display
                sample_dict = {}
                for key, value in sample.items():
                    if isinstance(value, dict) and 'bytes' in value:
                        sample_dict[key] = "<Image Data>"
                    else:
                        sample_dict[key] = value
                st.json(sample_dict)
    
    # Footer
    st.markdown("---")
    st.markdown("### Dataset Statistics")
    
    if 'categories' in locals():
        total_samples = 0
        category_stats = []
        
        for cat_name, files in categories.items():
            cat_samples = 0
            for file in files:
                try:
                    df_temp = pd.read_parquet(file)
                    cat_samples += len(df_temp)
                except:
                    pass
            total_samples += cat_samples
            category_stats.append({"Category": cat_name, "Samples": cat_samples, "Files": len(files)})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Categories", len(categories))
        with col2:
            st.metric("Total Files", sum(len(files) for files in categories.values()))
        with col3:
            st.metric("Total Samples (Estimated)", total_samples)
        
        # Show category breakdown
        with st.expander("Category Breakdown"):
            stats_df = pd.DataFrame(category_stats)
            st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    main()