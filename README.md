# HommageTools for ComfyUI

A comprehensive collection of utility nodes for ComfyUI that enhance your workflow with advanced text processing, image manipulation, and workflow control capabilities.

![GitHub](https://img.shields.io/github/license/ArtHommage/HommageTools)
![GitHub last commit](https://img.shields.io/github/last-commit/ArtHommage/HommageTools)

## 🚀 Features

### Text Processing
- **Regex Parser**: Parse text using regular expressions
- **Parameter Extractor**: Extract labeled parameters from text strings
- **Text Cleanup**: Comprehensive text cleanup with configurable rules

### Image Processing
- **Smart Resize**: Intelligent image resizing with aspect ratio preservation
- **Resolution Recommender**: Get optimal resolution suggestions for your workflow
- **Levels Correction**: Advanced image levels adjustment with reference matching
- **Training Size Calculator**: Calculate optimal dimensions for ML training
- **Base Shift Calculator**: Compute base shift values for images

### Workflow Control
- **Type Converter**: Flexible type conversion between string, integer, and float
- **Switch Node**: Simple workflow control mechanisms
- **Layer Management**: Create and export layered images in PSD/TIFF formats

## 🛠️ Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/ArtHommage/HommageTools.git
   ```

3. Install requirements:
   ```bash
   cd HommageTools
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## 📖 Node Documentation

### Regex Parser (HTRegexNode)
Process text using regular expressions
- **Inputs**:
  - Input Text: Text to be parsed
  - Regex Pattern: The regular expression pattern
- **Outputs**:
  - Parsed Text: Result of the regex operation

### Smart Resize (HTResizeNode)
Intelligently resize images while maintaining aspect ratios
- **Features**:
  - Multiple interpolation methods
  - Smart scaling options
  - Proper handling of both images and latents
  - Various cropping and padding options

### Resolution Recommender (HTResolutionNode)
Get optimal resolution suggestions based on various quality priorities
- **Features**:
  - Standard and custom resolution lists
  - Quality-based recommendations
  - Aspect ratio preservation
  - Detailed dimension analysis

[Documentation for other nodes follows similar pattern...]

## 🔧 Usage Examples

### Basic Text Processing
```python
# Example workflow using the Regex Parser
regex_node = HTRegexNode()
result = regex_node.parse_text(
    input_text="Sample text 123", 
    regex_pattern="\d+"
)
```

### Image Resizing
```python
# Example of smart image resizing
resize_node = HTResizeNode()
result = resize_node.resize_media(
    divisible_by="8",
    interpolation="bicubic",
    scaling_mode="short_side",
    crop_or_pad_mode="center",
    image=your_image_tensor
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the ComfyUI community for their support and feedback
- Special thanks to all contributors who have helped improve this toolkit