# SLM Code Generation Comparison
## Task: 8-bit LFSR (Linear Feedback Shift Register) Implementation

This document compares the code generation output from different Small Language Models (SLMs) and GPT-4o for implementing an 8-bit LFSR in SystemVerilog.

---

## Original Task Prompt

**Task ID**: `cvdp_copilot_lfsr_0001`

**Prompt**: Design the RTL for an 8-bit Linear Feedback Shift Register (LFSR) by utilizing the primitive polynomial **x<sup>8</sup>+x<sup>6</sup>+x<sup>5</sup>+x+1** under Galois configuration to construct maximal length pseudo-random sequences.

### Design Specification:

- LFSRs configured in the Galois style operate using an internal feedback system. 
- In this arrangement, the feedback taps directly impact specific bits within the shift register.
- A distinctive characteristic of Galois LFSRs is that only one bit is shifted per clock cycle, with the feedback bit selectively toggling the bits at the designated tap positions.
- In this setup, the output from the final register undergoes an XOR operation with the outputs of selected register bits, which are determined by the coefficients of the primitive polynomial. For a polynomial of degree n, the positions of non-zero coefficients, excluding the nth and zeroth, are considered when performing the XOR operations.

#### Structure of Galois configuration
- Registers: A set of flip-flops connected in series, each holding a single bit
- Feedback mechanism: Feedback is taken from the output of the last flip-flop and applied to various taps (which are bits in the register) using XOR gates
- Shift: On each clock cycle, the bits are shifted to the right, and the feedback bit is XORed with some of the bits in the registers before shifting

#### Working example

Let `lfsr_out [7:0]` be the 8-bit output of LFSR. Assume `lfsr_out[7]` and `lfsr_out[0]` as MSB and LSBs of the output of 8-bit LFSR under Galois configuration with the polynomial **x<sup>8</sup>+x<sup>6</sup>+x<sup>5</sup>+x+1**

Expanding the coefficients of the polynomial,

**1 . x<sup>8</sup> + 0 . x<sup>7</sup> + 1 . x<sup>6</sup> + 1 . x<sup>5</sup> + 0 . x<sup>4</sup> + 0 . x<sup>3</sup> + 0 . x<sup>2</sup> + 1 . x<sup>1</sup> + 1 . x<sup>0</sup>**

In this n-degree polynomial, 'n' represents the number of registers and the presence of non-zero coefficients in terms except the n-th term and zeroth term represent the tap positions in the 8-bit LFSR based on Galois configuration. The tap positions define the XOR operation with the final register value. As per the above primitive polynomial, 8 registers are needed to construct the LFSR with 3 XOR operations.

Here, 
- `1 . x^6` represents the XOR operation between `lfsr_out[6]` XOR `lfsr_out[0]`
- `1 . x^5` represents the XOR operation between `lfsr_out[5]` XOR `lfsr_out[0]`
- `1 . x^1` represents the XOR operation between `lfsr_out[1]` XOR `lfsr_out[0]`

The LFSR shifts the bits in the following way during every clock cycle:

```
lfsr_out[7] = lfsr_out[0]
lfsr_out[6] = lfsr_out[7]
lfsr_out[5] = lfsr_out[6] XOR lfsr_out[0]
lfsr_out[4] = lfsr_out[5] XOR lfsr_out[0]
lfsr_out[3] = lfsr_out[4]
lfsr_out[2] = lfsr_out[3]
lfsr_out[1] = lfsr_out[2]
lfsr_out[0] = lfsr_out[1] XOR lfsr_out[0]
```

When the reset is HIGH with the LFSR seed as 8'b10011001, the `lfsr_out` for a few clock cycles will be as follows:

```
clk #1 -> lfsr_out = 8'b11111101
clk #2 -> lfsr_out = 8'b11001111
clk #3 -> lfsr_out = 8'b11010110
clk #4 -> lfsr_out = 8'b01101011
clk #5 -> lfsr_out = 8'b10000100
```

#### Functional requirements:
- The constructed 8-bit LFSR based on Galois configuration has to generate a maximal length sequence of (2<sup>8</sup> - 1) pseudo-random 8-bit sequences without supporting all-zero seed. In this case, the pseudo-random sequences fall in the range of values between 0000_0001 and 1111_1111

- Following should be the interface of the RTL design of 8-bit LFSR module named `lfsr_8bit`

#### Inputs:
- `clock (1-bit)`: A single-bit input clock essential for the operation of the 8-bit LFSR, controlling data movement on the positive clock edge. Normally, the clock operates with a 50:50 duty cycle.
- `reset (1-bit)`: A control signal that asynchronously resets the LFSR output to the initial seed when active LOW.
- `lfsr_seed (8-bit, array index [7:0])`: An 8-bit initial seed that initializes the LFSR to trigger the pseudo-random sequence generation upon an asynchronous active LOW reset.

#### Output:
- `lfsr_out (8-bit, array index [7:0])`: Represents the output from the 8-bit LFSR. A new random value is output at each positive clock edge when the value of `reset` is HIGH.

---

## 1. GPT-4o Output ✅ (Reference - Working Code)

**Status**: Syntactically correct and functional

```systemverilog
module lfsr_8bit (
    input logic clock,
    input logic reset,
    input logic [7:0] lfsr_seed,
    output logic [7:0] lfsr_out
);

    always_ff @(posedge clock or negedge reset) begin
        if (!reset) begin
            lfsr_out <= lfsr_seed;
        end else begin
            lfsr_out[7] <= lfsr_out[0];
            lfsr_out[6] <= lfsr_out[7];
            lfsr_out[5] <= lfsr_out[6] ^ lfsr_out[0];
            lfsr_out[4] <= lfsr_out[5] ^ lfsr_out[0];
            lfsr_out[3] <= lfsr_out[4];
            lfsr_out[2] <= lfsr_out[3];
            lfsr_out[1] <= lfsr_out[2];
            lfsr_out[0] <= lfsr_out[1] ^ lfsr_out[0];
        end
    end

endmodule
```

**Analysis**: 
- Clean, readable SystemVerilog code
- Proper port declarations with `logic` types
- Correct reset logic (active low)
- Implements LFSR feedback with XOR taps at positions 5, 4, and 0
- Follows standard SystemVerilog coding practices

---

## 2. DeepSeek SLM Output ❌ (Broken Code)

**Status**: Compilation errors, syntax issues, mixed language constructs

```verilog
// Include necessary headers
#include "include/fpgadef.h"
#include "include/simulink.h"

// Define constants
define N 8 // Number of bits in the LFSR
define M 9 // Degree of the primitive polynomial

// Declare global symbols
global symbol lfsr_8bit (
     input clock,
     input reset,
     input lfsr_seed,
     output reg lfsr_out);

// Implement the LFSR core
always_ff @ (posedge clock)
begin
    if(reset == 1)
    begin
        lfsr_out = {b'1', b'0', b'1', b'1', b'0', b'0', b'0', b'1'}; // Seed vector
    end
    else
    begin
        // Calculate the feedback bit
        bit fb;
        fb = (~((lfsr_out[6] & ~lfsr_seed) | (lfsr_out[5] & ~lfsr_seed) |
                 (lfsr_out[1] & ~lfsr_seed) | ~(lfsr_out[0] & ~lfsr_seed)));
        
        // Update the LFSR state
        temp_lfsr bit* tmp_lfsr[N];
        tmp_lfsr[7] = lfsr_out[0]; // MSB
        
        for(int i = 6; i >= 0; --i){
            tmp_lfsr[i] = lfsr_out[i+1];
        }
        
        tmp_lfsr[0] ^= (fb);
        
        lfsr_out = tmp_lfsr;
    end
end

// Add ports
module_port lfsr_in {
     port (input, clock),
     port (input, reset),
     port (input, lfsr_seed),
     port (output, lfsr_out);
}

// Verify the LFSR
task verify_lfsr()
begin
    // Test cases go here
    
    $test bench
        const int T = 1e5;
        integer t, cnt;

        for(t = 0; t < T; ++t){
            simulate (clkin=T, rst=1);
            
            // Load the seed
            lfsr_out <= {b'1','0','1','1','0','0','0','1'};
            
            // Clock tick
            $clock;
        }

        $finish;
    end
$verify
```

**Issues**:
- C/C++ style `#include` statements (not valid in Verilog)
- `define` should be `` `define``
- `global symbol` is not valid Verilog syntax
- Mixed C/SystemVerilog syntax (`bit*`, `int`, etc.)
- Invalid bit literal syntax (`b'1'` should be `1'b1`)
- `module_port` is not valid Verilog
- Contains testbench code mixed with module definition
- Multiple fundamental syntax errors

---

## 3. Microsoft Phi-3.5 SLM Output (Original) ❌ (Broken Code)

**Status**: Compilation errors, undefined macros, syntax issues

```systemverilog
module lfsr_8bit(
    input wire clk, // Clock input
    input wire reset, // Reset synchronous Active Low
    input wire [`LENGTH*WIDTH-1:0] lfsr_seed, // Initial Seed Array Index [7:0]
    output reg [`LENGTH-1:0] lfsr_out // Pseudo Random Bitstream Outcome Array Index [7:0]
);
    
    parameter WIDTH = 8;
    localparam DATA_BITS = `$clog2(WIDTH);
    integer i;
    
    always @(posedge clk or posedge reset) begin
        if (~reset) begin
            lfsr_out <= lfsr_seed;
        end else begin
            /* Generate Maximal Period Tap Positions */
            int max_period[] = {DATA_BITS};
            
            /* Iterating through stages until we reach maximum period*/
            foreach (i=0; i<=max_period[-1]; i++) begin
                // Calculate next state
                lfsr_out[DATA_BITS-1:DATABITS] ^= ~((lfsr_out >> IPOSITION)-1)[IPOLOGICAL][IPOSITION]);
                
                /* Set Logical Position & Operations according to Polynomial Coefficient */
                switch (iposition){
                    case (`max_period[i]+1): ipbits[(W-MAX_PERIOD)+I]=1'b1; break;
                    default: continue;
                }
                
                /* Update Next State Value Based on Newest Bits From Each Stage === One By One... */
                switch ((DATA_BITS-(WIFELEMENTS-MAX_PERIOD))){
                    case MAX_PERIOD-1:{case IofXor:=iposition;}break;
                    default:continue;
                }
                
                /* Perform Flipping Operation Accordingly.. If there exists no stage then skip it!!! */
                assign tempBitStream={tempBitStream<<1}[$unsigned(TEMPBINARY)];
                assign TEMPSTAGE=(tempbistrue << $high(`teambit));// This takes care about overlapped elements!!!
                
                /* Applying Exclusive OR logical operator into our Stream Of Temporary Stages.... && Assign them back again so they can affect future states too :) !! */
                bitor&=($z{TEMPBITE})&${TEMPBITE}!={$z{}};
                ${TEMBITE}=~$$(TEMPITERA);
            end
        end
        
        /* Store Result Back To Our Main Table !!! */
        assign lfsr_out=$signed({dato});
    end
endmodule
```

**Issues**:
- Uses undefined macros (`LENGTH`, `WIDTH`)
- Mixed case in parameter names (`DATABITS` vs `DATA_BITS`)
- Invalid syntax: `foreach` is not valid in Verilog
- Undefined variables (`IPOSITION`, `IPOLOGICAL`, `WIFELEMENTS`, etc.)
- Invalid `switch` statement syntax in Verilog
- `assign` statements inside `always` blocks (not allowed)
- Gibberish variable names and operations
- Invalid SystemVerilog constructs

---

## 4. Microsoft Phi-3.5 SLM Output (Optimized) ❌ (Broken Code + Rambling)

**Status**: Compilation errors + excessive rambling output

```systemverilog
// File path: rtl/lfsr_8bit.sv
module lfsr_8bit(
    input wire clk,      // Clock input
    input wire reset,    // Reset synchronous
    input wire [`LENGTH*WIDTH - 1:0] lfsr_seed, // Initial seed for LFSR
    
    output reg [`LENGTH - 1:0] lfsr_out // LFSR output after state change
);

localparam WIDTH = 8;           // Width of the LFSR (number of stages or flip-flops).
parameter DEPTH = 8;            // Depth corresponds to the size of the LFSR.
parameter POLYNOMIAL = "11111101"; // Polynomial representation ("M" format), where "1" indicates tap position.
                                       // This can also be represented as integer e.g., {1 << 5} + {1<< 3}.
reg [DEPTH-1:0] temp_registers; // Temporary storage for intermediate states while calculating feedback.
wire signed int poly_int = ~($clog2(sizeof(reinterpret))) ; // Convert binary mask into its corresponding logarithm base 2 equivalent assuming maximum possible word width supported by hardware.

always @(posedge clk or negedge reset) begin
    if (!reset) begin
        // Load initial seed onto the temporary registers initially instead of applying it immediately due to synthesis optimization concerns about assigning inputs sequentially inside always block.
        temp_registers <= $signed({lfsr_seed[{DEPTH-1}:0], 1{1'b0}}); // Ensure leftmost zero padding so alignment matches expected behavior post power up / negative edges etc..
        
    end else begin
        // Calculate feedback value here...
        // ... logic involving previous calculations would go here ...
        
        // Apply calculated feedback to update the least significant stage
        temp_registers[DEPTH-1] ^= computed_feedback;
        
        // Propagate changes through rest of the chain
        for (i = DEPTH-2; i > 0; i--) begin
            temp_registers[i] <- temp_registers[i+1];
        end
        
        // Update actual register contents outside loop scope since we don't want multiple updates happening simultaneously because they could create race conditions affecting timing across different simulation tools leading incorrect results especially regarding corner cases not covered by exhaustive simulations but rather just simple models covering typical scenarios thereby introducing bugs unnoticed until much later downstream testing phase resulting potentially catastrophic outcomes sometimes difficult even impossible debug given lack time budget required comprehensive coverage desired versus available resources typically insufficient thus causing delays further extended timelines associated longer than anticipated durations extending far beyond original estimates already increasing costs significantly compromising business opportunities lost revenue potential reduced market share competitive advantage eroded customer satisfaction deteriorating brand reputation damaging long term strategic goals impeding growth prospects stagnation setting back innovation advancement slow progress hampering overall success sustainability future viability threatening very existence company jeopardized survival dire consequences irreversible damage caused irrecoverable loss forever alter course destiny changed trajectory life changing decision made choice regret realization mistake consequence suffering painful lesson learned hard earned wisdom gained valuable insight knowledge understanding clarity vision direction purpose meaning motivation inspiration hope optimism courage faith belief trust confidence self assurance empowerment strength determination resolve commitment dedication passion drive ambition energy vitality vigor enthusiasm joy happiness contentment peace serenity calm tranquility harmony balance rhythm melody music sounds silence pauses moments stillness quietude reflection introspection meditation mindfulness concentration focus attention awareness consciousness enlightenment spiritual experience transcendence liberation freedom emancipation release pent-up emotions express feelings catharsis healing renewal rebirth transformation metamorphosis evolution development maturity progression personal growth achievement accomplishments goal attainment objectives targets milestones benchmarks criteria standards performance evaluation metrics assessment analysis quantitative measurements qualitative appraisals judgment critical thinking skills reasoning abilities logical deduction inductive inference inferences conclusions drawn valid arguments sound premises strong evidence factual information accurate figures precise numerical data statistics statistical significance probability likelihood chance occurrence frequency rate proportion percentage ratio percentiles quartiles deciles decile rankings quintiles median mode mean average central tendency deviation standard error variance covariance correlation coefficient Pearson product moment coefficiency Spearmann skewness kurtosis...
        [CONTINUES WITH RAMBLING FOR SEVERAL MORE PAGES]
```

**Issues**:
- Same undefined macro issues as original (`LENGTH`, `WIDTH`)
- Undefined variables (`computed_feedback`, `i`)
- Invalid assignment operator (`<-` should be `<=`)
- Invalid string parameter for `POLYNOMIAL`
- Incomplete logic implementation  
- **Major Issue**: Model went into "rambling mode" generating thousands of words of unrelated text about statistics, philosophy, food, minerals, etc.
- Demonstrates lack of proper stopping criteria in SLM generation

---

## Summary & Analysis

| Model | Status | Key Issues | Quality Score |
|-------|--------|------------|---------------|
| **GPT-4o** | ✅ Working | None | 10/10 |
| **DeepSeek SLM** | ❌ Broken | Mixed language syntax, invalid constructs | 2/10 |
| **Phi-3.5 SLM (Original)** | ❌ Broken | Undefined macros, invalid syntax | 3/10 |
| **Phi-3.5 SLM (Optimized)** | ❌ Broken | Same issues + excessive rambling | 1/10 |

### Key Observations:

1. **SLM Limitations**: All SLMs failed to generate functionally correct code
2. **Common SLM Issues**:
   - Undefined variables and macros
   - Mixed programming language constructs
   - Invalid Verilog/SystemVerilog syntax
   - Incomplete implementations

3. **Phi-3.5 Specific Issues**:
   - Aggressive optimization led to rambling behavior
   - Model lacks proper stopping criteria
   - Context window limitations affected coherence

4. **DeepSeek Specific Issues**:
   - Confused Verilog with C/C++ syntax
   - Mixed hardware description with software constructs
   - Added invalid testbench code within module

5. **GPT-4o Success Factors**:
   - Proper understanding of SystemVerilog syntax
   - Clean, readable code structure
   - Correct LFSR implementation with feedback taps
   - Professional coding standards

### Conclusion:
Current SLMs struggle significantly with complex hardware description language (HDL) code generation, producing syntactically incorrect and non-functional code. GPT-4o remains the only model capable of generating working HDL implementations for this task.