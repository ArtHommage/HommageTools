import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.HTDynamicSwitchNode",
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "HTDynamicSwitchNode") {
            // Keep track of initialization state
            let initializing = false;
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                // Set initializing flag to prevent loops
                initializing = true;
                
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Add initial input if none exists
                if (!this.inputs || this.inputs.length === 0 || 
                    !this.inputs.some(input => input.name && input.name.startsWith("input"))) {
                    this.addInput("input1", "*");
                }
                
                // Reset initializing flag
                initializing = false;
                return r;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info, output) {
                // Don't process during initialization to avoid loops
                if (initializing) {
                    return onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                }
                
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                
                try {
                    if (type === LiteGraph.INPUT && connected) {
                        // Find highest input index
                        let maxInputNum = 0;
                        for (let i = 0; i < this.inputs.length; i++) {
                            const input = this.inputs[i];
                            if (input.name && input.name.startsWith("input")) {
                                const num = parseInt(input.name.substring(5));
                                if (!isNaN(num) && num > maxInputNum) {
                                    maxInputNum = num;
                                }
                            }
                        }
                        
                        // Add a new input if we connected to the highest one
                        const slotInput = this.inputs[index];
                        if (slotInput && slotInput.name === `input${maxInputNum}`) {
                            this.addInput(`input${maxInputNum + 1}`, "*");
                        }
                    }
                } catch (error) {
                    console.error("Error in HTDynamicSwitchNode onConnectionsChange:", error);
                }
                
                return r;
            };
            
            // Add handling for node configuration loading
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function(info) {
                initializing = true;
                const r = configure ? configure.apply(this, arguments) : undefined;
                
                // Ensure we have at least one input
                if (!this.inputs || this.inputs.length === 0 || 
                    !this.inputs.some(input => input.name && input.name.startsWith("input"))) {
                    this.addInput("input1", "*");
                }
                
                // Make sure we have one more input than the highest connected input
                let maxInputNum = 0;
                for (let i = 0; i < this.inputs.length; i++) {
                    const input = this.inputs[i];
                    if (input.name && input.name.startsWith("input")) {
                        const num = parseInt(input.name.substring(5));
                        if (!isNaN(num) && num > maxInputNum && input.link != null) {
                            maxInputNum = num;
                        }
                    }
                }
                
                // Add one more input slot for the next connection
                const nextInputExists = this.inputs.some(input => 
                    input.name && input.name === `input${maxInputNum + 1}`);
                    
                if (!nextInputExists && maxInputNum > 0) {
                    this.addInput(`input${maxInputNum + 1}`, "*");
                }
                
                initializing = false;
                return r;
            };
        }
    },
});