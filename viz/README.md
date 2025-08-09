# Zebra-CoT Dataset Visualization Tool

An interactive web application for exploring and visualizing the Zebra-CoT multimodal chain-of-thought reasoning dataset.

## Features

- **Category Browser**: Navigate through different reasoning categories (Physics, Chemistry, Chess, etc.)
- **Sample Viewer**: Browse individual samples with questions, reasoning traces, and answers
- **Image Display**: View problem and reasoning images associated with each sample
- **Interactive Navigation**: Jump to specific samples or explore randomly
- **Statistics Dashboard**: View dataset statistics and category breakdowns
- **Responsive Layout**: Optimized for both desktop and mobile viewing

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset is downloaded in the parent directory:
```bash
# Dataset should be in ../datasets/Zebra-CoT/
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Or with specific port:
```bash
streamlit run app.py --server.port 8501
```

## Navigation

1. **Sidebar**: 
   - Select dataset category
   - Choose specific file
   - Navigate samples by index
   - Random sample selection

2. **Main View Tabs**:
   - **Question Tab**: View the problem statement and associated images
   - **Reasoning Tab**: Explore the chain-of-thought reasoning process
   - **Answer Tab**: See the final answer
   - **Metadata Tab**: View sample statistics and raw data

## Dataset Structure

The app expects the Zebra-CoT dataset to be organized as:
```
../datasets/Zebra-CoT/
├── Scientific Reasoning - Physics/
│   ├── train-00000-of-00002.parquet
│   └── train-00001-of-00002.parquet
├── Visual Logic & Strategic Games - Chess/
│   ├── train-00000-of-00008.parquet
│   └── ...
└── ...
```

## Features in Detail

### Sample Navigation
- Use the number input to jump to specific samples
- Click "Random Sample" to explore randomly
- View sample count for context

### Image Viewing
- Problem images are displayed in the Question tab
- Reasoning images show the step-by-step visual thinking process
- Images are automatically resized for optimal viewing

### Text Display
- Long reasoning traces can be expanded/collapsed
- Color-coded sections for better readability:
  - Red background for questions
  - Gray background for reasoning
  - Green background for answers

### Statistics
- View total categories, files, and samples
- Category breakdown shows distribution across domains
- Sample metadata includes text lengths and image counts

## Troubleshooting

If the dataset is not found:
1. Ensure the dataset is downloaded to `../datasets/Zebra-CoT/`
2. Check that parquet files are present in category subfolders
3. Verify file permissions

For performance issues with large files:
- The app caches loaded data for faster navigation
- Clear cache with: `streamlit cache clear`

## Development

To extend the app:
1. Modify `app.py` for new features
2. Update `requirements.txt` for additional dependencies
3. Test with different dataset categories