# W8A8Linear Architecture - Strategy Pattern

## Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Strategy Interfaces                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────┐  ┌─────────────────────┐ │
│  │ ActivationQuantizationStrategy   │  │ WeightQuantization  │ │
│  │         (Protocol)               │  │   Strategy          │ │
│  ├──────────────────────────────────┤  │   (Protocol)        │ │
│  │ + quantize(x, n_bits)           │  ├─────────────────────┤ │
│  │ + name: str                     │  │ + quantize(w, bits) │ │
│  └──────────────────────────────────┘  │ + name: str         │ │
│                                         └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                    ▲                              ▲
                    │                              │
        ┌───────────┴────────────┐    ┌───────────┴────────────┐
        │                        │    │                        │
┌───────────────────────┐ ┌──────────────────────┐ ┌────────────────────┐
│ PerToken              │ │ PerTensor            │ │PerChannel          │
│ ActivationQuant       │ │ ActivationQuant      │ │WeightQuant         │
└───────────────────────┘ └──────────────────────┘ └────────────────────┘

┌───────────────────────┐ ┌──────────────────────┐ ┌────────────────────┐
│ PerGroup              │ │ DensityBased         │ │PerTensor           │
│ ActivationQuant       │ │ ActivationQuant      │ │WeightQuant         │
└───────────────────────┘ └──────────────────────┘ └────────────────────┘

                                                     ┌────────────────────┐
                                                     │PerGroup            │
                                                     │WeightQuant         │
                                                     └────────────────────┘

                                                     ┌────────────────────┐
                                                     │SelectiveChannel    │
                                                     │WeightQuant         │
                                                     └────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Main Class                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│                     ┌──────────────────┐                        │
│                     │   W8A8Linear     │                        │
│                     ├──────────────────┤                        │
│                     │ - activation_    │◆───────────────────┐  │
│                     │   strategy       │                     │  │
│                     │ - weight         │                     │  │
│                     │ - bias           │                     │  │
│                     │ - n_bits_w       │      Uses           │  │
│                     │ - n_bits_a       │   (Composition)     │  │
│                     ├──────────────────┤                     │  │
│                     │ + forward(x)     │                     │  │
│                     │ + quantize_      │                     │  │
│                     │   activation(x)  │─────────────────────┘  │
│                     │ + from_float()   │                        │
│                     └──────────────────┘                        │
│                            ▲                                     │
│                            │                                     │
│                   Inherits │                                     │
│         ┌──────────────────┼────────────────────┐              │
│         │                  │                    │              │
│  ┌──────────────┐ ┌────────────────┐ ┌─────────────────┐     │
│  │W8A8Linear    │ │W8A8Linear      │ │W8A8Linear       │     │
│  │PerChannel    │ │PerTensor       │ │PerGroup         │     │
│  │(Legacy)      │ │(Legacy)        │ │(Legacy)         │     │
│  └──────────────┘ └────────────────┘ └─────────────────┘     │
│                                                                 │
│  ┌──────────────┐ ┌────────────────┐                         │
│  │W8A8Linear    │ │W8A8Linear      │                         │
│  │DensityBased  │ │SelectiveChannel│                         │
│  │(Legacy)      │ │(Legacy)        │                         │
│  └──────────────┘ └────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Relationships

### Strategy Pattern Flow

1. **W8A8Linear** holds a reference to an `activation_strategy`
2. When `quantize_activation(x)` is called, it delegates to `activation_strategy.quantize(x, n_bits)`
3. Different strategies implement different quantization algorithms
4. Strategies are interchangeable at runtime

### Weight Quantization Flow

1. Weight quantization happens during initialization via `from_float()`
2. A `WeightQuantizationStrategy` is selected based on `weight_quant` parameter
3. The strategy's `quantize()` method is called to quantize weights
4. The strategy name is stored in `weight_quant_name` for metadata

### Factory Method Pattern

```
from_float()
    │
    ├──> Create activation strategy based on act_quant parameter
    │
    ├──> Create W8A8Linear with activation strategy
    │
    ├──> Create weight strategy based on weight_quant parameter
    │
    ├──> Apply weight quantization using strategy
    │
    └──> Return quantized W8A8Linear instance
```

## Execution Flow

```
User Code
    │
    ├──> Creates float linear: nn.Linear(256, 512)
    │
    ├──> Calls W8A8Linear.from_float(
    │        module=float_linear,
    │        weight_quant="per_channel",
    │        act_quant="per_token"
    │    )
    │
    ├──> Factory creates PerTokenActivationQuantization strategy
    │
    ├──> Factory creates W8A8Linear with strategy
    │
    ├──> Factory creates PerChannelWeightQuantization strategy
    │
    ├──> Factory applies weight quantization
    │
    └──> Returns quantized W8A8Linear
         │
         ├──> User calls: quant_linear(input_tensor)
         │
         ├──> forward() is called
         │
         ├──> quantize_activation(x) delegates to strategy
         │    │
         │    └──> PerTokenActivationQuantization.quantize(x, 8)
         │
         ├──> F.linear(q_x, weight, bias)
         │
         └──> Returns quantized output
```

## Key Design Principles

### 1. Single Responsibility
- Each strategy class has ONE job: implement a specific quantization algorithm
- W8A8Linear has ONE job: manage the linear layer and delegate quantization

### 2. Open/Closed Principle
- Open for extension: New strategies can be added without modifying W8A8Linear
- Closed for modification: W8A8Linear doesn't need changes when adding strategies

### 3. Dependency Inversion
- W8A8Linear depends on the `ActivationQuantizationStrategy` protocol (abstraction)
- Not on concrete strategy implementations

### 4. Composition over Inheritance
- W8A8Linear **has a** strategy (composition)
- Instead of subclasses that **are** specialized linear layers (inheritance)

## Benefits Visualization

```
Before (Inheritance):
    W8A8Linear (abstract)
         ├─ W8A8LinearPerChannel
         ├─ W8A8LinearPerTensor
         ├─ W8A8LinearPerGroup
         ├─ W8A8LinearDensityBased
         └─ W8A8LinearSelectiveChannel

    Problem: 5 classes, tight coupling, hard to mix strategies

After (Strategy Pattern):
    W8A8Linear (uses strategies)
         │
         ├─ Activation: [PerToken | PerTensor | PerGroup | DensityBased]
         └─ Weight: [PerChannel | PerTensor | PerGroup | SelectiveChannel]

    Benefit: 1 class + 8 strategies, loose coupling, easy to combine
```

## Adding New Strategies

To add a new quantization strategy:

```python
# 1. Create a new strategy class
class MyNewActivationQuantization:
    @property
    def name(self) -> str:
        return "my_new_quant"

    def quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        # Your quantization logic here
        return quantized_x

# 2. Use it directly
linear = W8A8Linear(
    256, 512,
    activation_strategy=MyNewActivationQuantization()
)

# 3. Or add it to from_float() factory
# (just add an elif branch in the factory method)
```

No need to create a new subclass of W8A8Linear!
