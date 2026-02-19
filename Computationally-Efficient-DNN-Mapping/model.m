Model:
  # Actor Network Layers
  - name: "Actor_FC1"
    Dimensions:
      - K: 20      # Number of output neurons
      - C: 13      # Number of input features (based on the example state size)
    Precision: 32  # Assuming 32-bit floats
    Type: "FullyConnected"

  - name: "Actor_FC2"
    Dimensions:
      - K: 20      # Number of output neurons
      - C: 20      # Number of input neurons from previous layer
    Precision: 32
    Type: "FullyConnected"

  - name: "Actor_Output"
    Dimensions:
      - K: 3       # Number of output actions (3 in this case: -1, 0, 1)
      - C: 20      # Number of input neurons from previous layer
    Precision: 32
    Type: "FullyConnected"

  # Critic Network Layers
  - name: "Critic_FC1"
    Dimensions:
      - K: 20
      - C: 13
    Precision: 32
    Type: "FullyConnected"

  - name: "Critic_FC2"
    Dimensions:
      - K: 20
      - C: 20
    Precision: 32
    Type: "FullyConnected"

  - name: "Critic_Output"
    Dimensions:
      - K: 1       # Final output for Critic network
      - C: 20
    Precision: 32
    Type: "FullyConnected"

System:
  PEArray:
    Size: 256      # Number of processing elements
    DataPrecision: 32
    RF:
      Size: 672    # Register file size in bytes
      ReadBW: 64   # Register file read bandwidth in bits/cycle
      WriteBW: 64  # Register file write bandwidth in bits/cycle
    MAC:
      Energy: 0.0375 pJ  # Energy per MAC operation

  SharedMemory:
    L2Cache:
      Size: 108K   # L2 Cache size in bytes
      ReadBW: 128  # Read bandwidth in bits/cycle
      WriteBW: 128 # Write bandwidth in bits/cycle
    DRAM:
      Size: 400M   # DRAM size in bytes
      ReadBW: 64   # DRAM read bandwidth in bits/cycle
      WriteBW: 64  # DRAM write bandwidth in bits/cycle