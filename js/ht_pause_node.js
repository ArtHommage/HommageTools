/**
 * File: js/ht_pause_node.js
 * 
 * UI Component for HommageTools Pause Node
 * Handles the pause button and visual state management
 */

// Safer app import with error handling
let app;
try {
    const appModule = await import('/scripts/app.js');
    app = appModule.app;
} catch (error) {
    console.error('Failed to load ComfyUI app module:', error);
    // Provide fallback or error state
    app = {
        registerExtension: () => console.warn('ComfyUI app not available')
    };
}

// Register node behavior when ComfyUI loads
app.registerExtension({
    name: "HommageTools.PauseNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only modify our specific node
        if (!nodeData?.name || nodeData.name !== "HTPauseNode") {
            return;
        }

        try {
            // Add custom widget to node
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                try {
                    // Add resume button widget
                    this.addWidget(
                        "button", 
                        "Resume", 
                        null, 
                        () => {
                            // Send resume signal to backend
                            const workflow = app.graph?._nodes.find(n => n.id === this.id);
                            if (workflow) {
                                app.queuePrompt(0, workflow.id);
                            }
                        }
                    );

                    // Initialize paused state
                    this.paused = true;
                    this.pauseTitle = this.properties?.pause_title || "Paused";
                } catch (error) {
                    console.error('Error in node creation:', error);
                }

                return result;
            };

            // Override drawing to show pause state
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                try {
                    if (onDrawForeground) {
                        onDrawForeground.apply(this, arguments);
                    }

                    if (this.paused) {
                        // Draw pause indicator
                        const text = this.pauseTitle || "Paused";
                        ctx.save();
                        ctx.font = "bold 14px Arial";
                        ctx.fillStyle = "#FF5555";
                        ctx.textAlign = "center";
                        ctx.fillText(text, this.size[0] * 0.5, this.size[1] * 0.5);
                        ctx.restore();
                    }
                } catch (error) {
                    console.error('Error in draw:', error);
                }
            };

            // Handle execution state changes
            nodeType.prototype.onExecuted = function(message) {
                this.paused = false;
                this.setDirtyCanvas(true);
            };

            nodeType.prototype.onExecutionStart = function() {
                this.paused = true;
                this.setDirtyCanvas(true);
            };
        } catch (error) {
            console.error('Error in node registration:', error);
        }
    }
});