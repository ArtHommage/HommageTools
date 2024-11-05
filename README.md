# HommageTools for ComfyUI

A collection of utility nodes for ComfyUI that provide additional functionality for text processing and image manipulation.

## Features

- **Regex Parser**: Parse text using regular expressions
- **Smart Resize**: Resize images with intelligent dimension handling
- **Resolution Recommender**: Get optimal resolution suggestions
- **Pause Workflow**: Add workflow control points
- **Type Converter**: Convert between data types
- **Switch**: Simple workflow control switch
- **File Queue**: Manage batches of files

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/HommageTools.git
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## Node Documentation

### Regex Parser
Performs regular expression parsing on input text
- **Inputs**:
  - Input Text: The text to be parsed
  - Regex Pattern: The regular expression pattern to apply
- **Outputs**:
  - Parsed Text: The result of the regex operation

[Documentation for other nodes follows similar pattern...]

## Development

To add new nodes:
1. Create a new file in the `nodes` directory
2. Implement your node class
3. Add the node to the mappings in `__init__.py`

## License

[Your chosen license]