// File: homage_tools/web/js/ht_dynamic_switch_node.js
// Version: 1.1.0
// Description: UI implementation for HTDynamicSwitchNode with dynamic input handling

(function() {
    // Wait for Comfy UI to be ready
    const waitForComfy = setInterval(function() {
        if (window.LiteGraph && window.app) {
            clearInterval(waitForComfy);
            registerHTDynamicSwitchNode();
        }
    }, 200);
    
    function registerHTDynamicSwitchNode() {
        // Register with ComfyUI's extension system if available
        if (window.app?.registerExtension) {
            window.app.registerExtension({
                name: "HT.DynamicSwitch",
                async beforeRegisterNodeDef(nodeType, nodeData) {
                    if (nodeData.name === "HTDynamicSwitchNode") {
                        // Add onConnectionsChange handler for dynamic inputs
                        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
                        nodeType.prototype.onNodeCreated = function() {
                            if (origOnNodeCreated) {
                                origOnNodeCreated.apply(this, arguments);
                            }
                            
                            // Initialize dynamic input management
                            this.onConnectionsChange = function(type, index, connected, link_info) {
                                // Only handle input connections (type === LiteGraph.INPUT)
                                if (type === 0) {
                                    // Find inputs that start with "input"
                                    const inputKeys = Object.keys(this.inputs)
                                        .filter(key => key.startsWith("input"))
                                        .map(key => parseInt(key.substring(5)))
                                        .filter(key => !isNaN(key));
                                    
                                    // Find the highest input number
                                    const maxInputNumber = inputKeys.length > 0 ? Math.max(...inputKeys) : 0;
                                    
                                    if (connected) {
                                        // When a connection is made, add a new input slot if needed
                                        const nextInputNumber = maxInputNumber + 1;
                                        this.addInput(`input${nextInputNumber}`, "*");
                                    }
                                }
                            };
                        };
                        
                        // Add custom context menu for quick selection
                        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                        nodeType.prototype.getExtraMenuOptions = function(_, options) {
                            if (origGetExtraMenuOptions) {
                                origGetExtraMenuOptions.apply(this, arguments);
                            }
                            
                            // Add input selection options to context menu
                            const inputOptions = [];
                            
                            for (let i = 0; i < this.inputs.length; i++) {
                                const input = this.inputs[i];
                                if (input.name.startsWith("input")) {
                                    const inputNumber = parseInt(input.name.substring(5));
                                    if (!isNaN(inputNumber)) {
                                        inputOptions.push({
                                            content: `Select Input ${inputNumber}`,
                                            callback: () => {
                                                // Find and update the select widget
                                                for (const widget of this.widgets) {
                                                    if (widget.name === "select") {
                                                        widget.value = inputNumber;
                                                        widget.callback(inputNumber);
                                                        break;
                                                    }
                                                }
                                            }
                                        });
                                    }
                                }
                            }
                            
                            // Add submenu if we have input options
                            if (inputOptions.length > 0) {
                                options.push({
                                    content: "Select Input...",
                                    submenu: inputOptions
                                });
                            }
                        };
                    }
                }
            });
        }
    }
})();