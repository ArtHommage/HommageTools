// File: homage_tools/web/js/ht_seed_advanced_ui.js
// Version: 1.0.0
// Description: Enhanced UI for HTSeedAdvancedNode with interactive controls

(function() {
    // Wait for Comfy UI to be ready
    const waitForComfy = setInterval(function() {
        if (window.LiteGraph && window.app) {
            clearInterval(waitForComfy);
            registerHTSeedAdvancedUI();
        }
    }, 200);

    function registerHTSeedAdvancedUI() {
        // Define custom widget for the seed input
        class SeedAdvancedWidget extends window.ComfyWidgets.INT {
            constructor(node, inputName, inputData, app) {
                // Call parent constructor
                super(node, inputName, inputData, app);
                
                // Store original values
                this.seed = this.value;
                this.random = false;
                this.lastChangedSeed = 0;
                
                // Only apply to the seed input
                if (inputName !== "seed") {
                    return;
                }
                
                // Create container for custom controls
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.alignItems = "center";
                container.style.gap = "5px";
                
                // Move the original widget to our container
                const widgetEl = this.widget.element;
                widgetEl.parentElement.appendChild(container);
                container.appendChild(widgetEl);
                widgetEl.style.width = "calc(100% - 80px)";
                
                // Add a random button
                const randomBtn = document.createElement("button");
                randomBtn.textContent = "🎲";
                randomBtn.title = "Generate random seed";
                randomBtn.style.width = "24px";
                randomBtn.style.height = "24px";
                randomBtn.style.padding = "0";
                randomBtn.style.fontSize = "16px";
                randomBtn.style.cursor = "pointer";
                
                // Add event listener for random button
                randomBtn.addEventListener("click", () => {
                    const randomSeed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
                    this.value = randomSeed;
                    this.widget.value = randomSeed;
                    this.widget.callback(randomSeed);
                    this.updateNodeTitle(randomSeed);
                    this.lastChangedSeed = Date.now();
                });
                
                container.appendChild(randomBtn);
                
                // Add a toggle button for auto random
                const autoRandomBtn = document.createElement("button");
                autoRandomBtn.textContent = "🔄";
                autoRandomBtn.title = "Toggle auto random seed";
                autoRandomBtn.style.width = "24px";
                autoRandomBtn.style.height = "24px";
                autoRandomBtn.style.padding = "0";
                autoRandomBtn.style.fontSize = "16px";
                autoRandomBtn.style.cursor = "pointer";
                
                // Add event listener for auto random toggle
                autoRandomBtn.addEventListener("click", () => {
                    this.random = !this.random;
                    autoRandomBtn.style.backgroundColor = this.random ? "#44A" : "";
                    
                    // Find and set the random_seed input
                    for (const [key, input] of Object.entries(this.node.inputs)) {
                        if (input.name === "random_seed") {
                            this.node.setProperty(`${key}__value`, this.random);
                            break;
                        }
                    }
                });
                
                container.appendChild(autoRandomBtn);
                
                // Add a copy button
                const copyBtn = document.createElement("button");
                copyBtn.textContent = "📋";
                copyBtn.title = "Copy seed to clipboard";
                copyBtn.style.width = "24px";
                copyBtn.style.height = "24px";
                copyBtn.style.padding = "0";
                copyBtn.style.fontSize = "16px";
                copyBtn.style.cursor = "pointer";
                
                // Add event listener for copy button
                copyBtn.addEventListener("click", () => {
                    navigator.clipboard.writeText(this.value.toString())
                        .then(() => {
                            // Show feedback
                            const originalColor = copyBtn.style.backgroundColor;
                            copyBtn.style.backgroundColor = "#6a6";
                            setTimeout(() => {
                                copyBtn.style.backgroundColor = originalColor;
                            }, 500);
                        })
                        .catch(err => {
                            console.error('Failed to copy seed: ', err);
                        });
                });
                
                container.appendChild(copyBtn);
                
                // Store references
                this.element = container;
                this.autoRandomBtn = autoRandomBtn;
                this.randomBtn = randomBtn;
                this.copyBtn = copyBtn;
                
                // Set initial node title
                this.updateNodeTitle(this.value);
                
                // Listen for seed updates from server
                if (app.ws) {
                    app.ws.addEventListener("message", (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (data?.type === "ht-seed-update" && data?.data?.node_id === this.node.id) {
                                const newSeed = data.data.seed;
                                if (newSeed !== this.value) {
                                    this.value = newSeed;
                                    this.widget.value = newSeed;
                                    this.updateNodeTitle(newSeed);
                                }
                            }
                        } catch (error) {
                            // Ignore parsing errors
                        }
                    });
                }
            }
            
            updateNodeTitle(seedValue) {
                if (!this.node.title || this.node.title === this.node.type) {
                    this.node.title = `${this.node.type}`;
                }
                
                // Get additional info from widgets if available
                let mode = "";
                let iterValue = "";
                
                for (const w of this.node.widgets) {
                    if (w.name === "seed_mode") {
                        mode = w.value;
                    } else if (w.name === "iter_value") {
                        iterValue = w.value;
                    }
                }
                
                // Create title with mode info if available
                let modeInfo = "";
                if (mode === "iter_add") {
                    modeInfo = ` (+${iterValue})`;
                } else if (mode === "iter_mult") {
                    modeInfo = ` (×${iterValue/1000})`;
                } else if (mode === "derived") {
                    modeInfo = " (derived)";
                }
                
                let newTitle = `${this.node.type} (${seedValue})${modeInfo}`;
                
                if (this.node.title !== newTitle) {
                    this.node.title = newTitle;
                    this.node.setDirtyCanvas(true, true);
                }
            }
        }

        // Register extension
        if (window.app?.registerExtension) {
            window.app.registerExtension({
                name: "HT.SeedAdvanced",
                async beforeRegisterNodeDef(nodeType, nodeData) {
                    if (nodeData.name === "HTSeedAdvancedNode") {
                        // Save the original INT widget
                        const originalINTWidget = window.ComfyWidgets["INT"];
                        
                        // Add widget customization
                        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
                        nodeType.prototype.onNodeCreated = function() {
                            if (origOnNodeCreated) {
                                origOnNodeCreated.apply(this, arguments);
                            }
                            
                            // Apply custom widget
                            this.widgets_start_y = 10;
                            window.ComfyWidgets["INT"] = SeedAdvancedWidget;
                            
                            // Restore default widget after widget creation
                            setTimeout(() => {
                                window.ComfyWidgets["INT"] = originalINTWidget;
                            }, 50);
                        };
                        
                        // Add custom context menu
                        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                        nodeType.prototype.getExtraMenuOptions = function(_, options) {
                            if (origGetExtraMenuOptions) {
                                origGetExtraMenuOptions.apply(this, arguments);
                            }
                            
                            // Add seed operations to context menu
                            options.push(
                                {
                                    content: "➕ Increment Seed",
                                    callback: () => {
                                        for (const w of this.widgets) {
                                            if (w.name === "seed") {
                                                const newSeed = (w.value + 1) % Number.MAX_SAFE_INTEGER;
                                                w.value = newSeed;
                                                w.callback(newSeed);
                                                break;
                                            }
                                        }
                                    }
                                },
                                {
                                    content: "➖ Decrement Seed",
                                    callback: () => {
                                        for (const w of this.widgets) {
                                            if (w.name === "seed") {
                                                const newSeed = (w.value - 1 + Number.MAX_SAFE_INTEGER) % Number.MAX_SAFE_INTEGER;
                                                w.value = newSeed;
                                                w.callback(newSeed);
                                                break;
                                            }
                                        }
                                    }
                                },
                                {
                                    content: "🎲 Randomize Seed",
                                    callback: () => {
                                        for (const w of this.widgets) {
                                            if (w.name === "seed") {
                                                const randomSeed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
                                                w.value = randomSeed;
                                                w.callback(randomSeed);
                                                break;
                                            }
                                        }
                                    }
                                },
                                {
                                    content: "🔄 Toggle Auto Random",
                                    callback: () => {
                                        for (const w of this.widgets) {
                                            if (w.name === "random_seed") {
                                                w.value = !w.value;
                                                w.callback(w.value);
                                                break;
                                            }
                                        }
                                    }
                                }
                            );
                        };
                        
                        // Add seed presets
                        const origOnSubgraphNewOptions = nodeType.prototype.onSubgraphNewOptions;
                        nodeType.prototype.onSubgraphNewOptions = function(options) {
                            if (origOnSubgraphNewOptions) {
                                origOnSubgraphNewOptions.apply(this, arguments);
                            }
                            
                            // Add helpful preset combinations
                            options.unshift(
                                {
                                    title: "📈 Sequential Seeds",
                                    callback: () => {
                                        for (const w of this.widgets) {
                                            if (w.name === "seed_mode") {
                                                w.value = "iter_add";
                                                w.callback("iter_add");
                                            } else if (w.name === "iter_value") {
                                                w.value = 1;
                                                w.callback(1);
                                            }
                                        }
                                    }
                                },
                                {
                                    title: "🔀 Derived Seeds",
                                    callback: () => {
                                        for (const w of this.widgets) {
                                            if (w.name === "seed_mode") {
                                                w.value = "derived";
                                                w.callback("derived");
                                            }
                                        }
                                    }
                                }
                            );
                        };
                    }
                }
            });
        }
    }
})();