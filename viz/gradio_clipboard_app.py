#!/usr/bin/env python3
"""
Gradio Clipboard Image Manager
Paste images from clipboard and save them to test_images folder
"""

import gradio as gr
from PIL import Image
import os
from datetime import datetime
import shutil
from pathlib import Path
import numpy as np

# Create test_images folder if it doesn't exist
TEST_IMAGES_DIR = Path("test_images")
TEST_IMAGES_DIR.mkdir(exist_ok=True)

# Global counter for naming
image_counter = len(list(TEST_IMAGES_DIR.glob("*.png")))

def get_saved_images():
    """Get list of saved images"""
    images = []
    for img_path in sorted(TEST_IMAGES_DIR.glob("*.png"))[::-1]:  # Most recent first
        images.append(str(img_path))
    return images[:20]  # Show last 20 images

def save_image(image, custom_name=None):
    """Save the pasted image to test_images folder"""
    global image_counter
    
    if image is None:
        return None, "No image provided", get_saved_images()
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            return None, "Invalid image format", get_saved_images()
        
        # Generate filename
        if custom_name and custom_name.strip():
            # Use custom name
            filename = custom_name.strip()
            if not filename.endswith('.png'):
                filename += '.png'
        else:
            # Auto-generate name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_counter += 1
            filename = f"image_{timestamp}_{image_counter:04d}.png"
        
        # Save image
        save_path = TEST_IMAGES_DIR / filename
        image.save(save_path, "PNG")
        
        # Get file size
        file_size = os.path.getsize(save_path) / 1024  # in KB
        
        success_msg = f"✅ Saved: {filename} ({image.size[0]}x{image.size[1]}, {file_size:.1f}KB)"
        
        return image, success_msg, get_saved_images()
        
    except Exception as e:
        return None, f"❌ Error saving image: {str(e)}", get_saved_images()

def delete_image(image_path):
    """Delete a saved image"""
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            return f"✅ Deleted: {os.path.basename(image_path)}", get_saved_images()
        except Exception as e:
            return f"❌ Error deleting: {str(e)}", get_saved_images()
    return "No image selected", get_saved_images()

def clear_all_images():
    """Clear all images from test_images folder"""
    try:
        count = 0
        for img_path in TEST_IMAGES_DIR.glob("*.png"):
            img_path.unlink()
            count += 1
        return f"✅ Cleared {count} images", []
    except Exception as e:
        return f"❌ Error clearing images: {str(e)}", get_saved_images()

def get_folder_stats():
    """Get statistics about the test_images folder"""
    image_files = list(TEST_IMAGES_DIR.glob("*.png"))
    total_size = sum(os.path.getsize(f) for f in image_files) / (1024 * 1024)  # in MB
    
    stats = f"""
    📁 Folder: {TEST_IMAGES_DIR.absolute()}
    🖼️ Total Images: {len(image_files)}
    💾 Total Size: {total_size:.2f} MB
    """
    return stats.strip()

def batch_save_images(images):
    """Save multiple images at once"""
    if not images:
        return "No images provided", get_saved_images()
    
    saved_count = 0
    failed_count = 0
    
    for idx, image in enumerate(images):
        if image is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_{timestamp}_{idx+1:03d}.png"
                
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                save_path = TEST_IMAGES_DIR / filename
                image.save(save_path, "PNG")
                saved_count += 1
            except:
                failed_count += 1
    
    msg = f"✅ Saved {saved_count} images"
    if failed_count > 0:
        msg += f", ❌ Failed: {failed_count}"
    
    return msg, get_saved_images()

# Create Gradio interface
with gr.Blocks(title="Clipboard Image Manager", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📋 Clipboard Image Manager
    
    Paste images from your clipboard and save them to the `test_images` folder.
    
    **How to use:**
    1. Copy an image to clipboard (Ctrl+C / Cmd+C)
    2. Click in the image area below and paste (Ctrl+V / Cmd+V)
    3. Optionally add a custom name
    4. Click Save to store the image
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Main image input
            with gr.Group():
                gr.Markdown("### 📥 Paste Image Here")
                image_input = gr.Image(
                    label="Paste or Upload Image",
                    type="pil",
                    height=400,
                    elem_id="image_input"
                )
                
                with gr.Row():
                    custom_name_input = gr.Textbox(
                        label="Custom Name (optional)",
                        placeholder="Enter filename (without extension)",
                        scale=3
                    )
                    save_btn = gr.Button("💾 Save Image", variant="primary", scale=1)
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
            
            # Batch upload section
            with gr.Group():
                gr.Markdown("### 📦 Batch Upload")
                batch_input = gr.File(
                    label="Drop multiple images here",
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_save_btn = gr.Button("💾 Save All", variant="primary")
                batch_status = gr.Textbox(
                    label="Batch Status",
                    interactive=False,
                    lines=1
                )
        
        with gr.Column(scale=1):
            # Folder stats
            with gr.Group():
                gr.Markdown("### 📊 Folder Statistics")
                stats_text = gr.Textbox(
                    value=get_folder_stats(),
                    label="",
                    interactive=False,
                    lines=3
                )
                refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")
            
            # Saved images gallery
            with gr.Group():
                gr.Markdown("### 🖼️ Recent Images")
                gallery = gr.Gallery(
                    value=get_saved_images(),
                    label="Saved Images",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    height=300,
                    object_fit="contain"
                )
                
                with gr.Row():
                    selected_image = gr.Textbox(
                        label="Selected Image",
                        interactive=False,
                        visible=False
                    )
                    delete_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop")
                    clear_all_btn = gr.Button("🗑️ Clear All", size="sm", variant="stop")
    
    # JavaScript for better clipboard handling
    gr.HTML("""
    <script>
    // Enhanced clipboard paste handling
    document.addEventListener('paste', function(e) {
        const imageInput = document.querySelector('#image_input canvas, #image_input img');
        if (imageInput && e.clipboardData && e.clipboardData.items) {
            for (let item of e.clipboardData.items) {
                if (item.type.indexOf('image') !== -1) {
                    console.log('Image detected in clipboard');
                }
            }
        }
    });
    
    // Auto-focus on image input
    window.addEventListener('load', function() {
        const imageArea = document.querySelector('#image_input');
        if (imageArea) {
            imageArea.setAttribute('tabindex', '0');
        }
    });
    </script>
    """)
    
    # Event handlers
    save_btn.click(
        fn=save_image,
        inputs=[image_input, custom_name_input],
        outputs=[image_input, status_text, gallery]
    ).then(
        fn=get_folder_stats,
        outputs=stats_text
    )
    
    def handle_batch_save(files):
        if not files:
            return "No files provided", get_saved_images()
        
        images = []
        for file in files:
            try:
                img = Image.open(file.name)
                images.append(img)
            except:
                pass
        
        return batch_save_images(images)
    
    batch_save_btn.click(
        fn=handle_batch_save,
        inputs=batch_input,
        outputs=[batch_status, gallery]
    ).then(
        fn=get_folder_stats,
        outputs=stats_text
    )
    
    def select_image(evt: gr.SelectData):
        if evt.value:
            return evt.value['image']['path']
        return None
    
    gallery.select(
        fn=select_image,
        outputs=selected_image
    )
    
    delete_btn.click(
        fn=delete_image,
        inputs=selected_image,
        outputs=[status_text, gallery]
    ).then(
        fn=get_folder_stats,
        outputs=stats_text
    )
    
    clear_all_btn.click(
        fn=clear_all_images,
        outputs=[status_text, gallery]
    ).then(
        fn=get_folder_stats,
        outputs=stats_text
    )
    
    refresh_btn.click(
        fn=get_folder_stats,
        outputs=stats_text
    ).then(
        fn=get_saved_images,
        outputs=gallery
    )
    
    # Load initial data
    app.load(
        fn=get_saved_images,
        outputs=gallery
    )

# Custom CSS for better styling
app.css = """
#image_input {
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    transition: border-color 0.3s;
}

#image_input:hover {
    border-color: #4f46e5;
}

#image_input:focus {
    outline: none;
    border-color: #4f46e5;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
}

#gallery {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px;
}

.gradio-button-primary {
    background: linear-gradient(to right, #4f46e5, #6366f1);
}

.gradio-button-primary:hover {
    background: linear-gradient(to right, #4338ca, #4f46e5);
}
"""

if __name__ == "__main__":
    print(f"📁 Images will be saved to: {TEST_IMAGES_DIR.absolute()}")
    print("🚀 Starting Gradio app...")
    print("📋 Copy an image and paste it in the app!")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )