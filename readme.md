# MaRCoS Client (Modified for Multi-Frequency Settings)

This repository is a fork of [vnegnev/marcos_client](https://github.com/vnegnev/marcos_client), extending its functionality to handle multi-frequency settings in MRI experiments. 

## Main Modifications  

The original `experiment.py` lacked support for multi-frequency configurations, which meant only a single carrier frequency could be set during a sequence. This limitation made it unsuitable for applications such as multi-slice 2D acquisitions.  

The FPGA backend supports dynamic updates to frequency (`lo0_freq`, `lo1_freq`, `lo2_freq`) and resets (`lo0_rst`, `lo1_rst`, `lo2_rst`) at specific sequence nodes. To leverage this capability, this fork introduces a new file, `experiment_multifreq.py`, which includes:  
- A reimplementation of the `flo2int()` functions, enabling frequency adjustments during sequence execution.  
- Enhanced support for complex multi-slice MRI acquisitions.

## Gradient Board Initialization Improvement

In earlier versions, when initializing the gradient board (OCRA1), the firmware would immediately enable the DAC output right after a RESET. This sequence caused the AD5781 DAC to output a transient -10V voltage after RESET, which could lead to saturation or abnormal behavior in the gradient amplifier module until a zero value was explicitly written to the DAC register.

In this update, the initialization process has been refined:
- After a RESET, the DAC output is kept disabled.
- Only after writing a zero value to the DAC register is the output enabled.
- This ensures the output voltage remains at 0V, effectively preventing transient spikes and avoiding potential damage or malfunction of the gradient amplification system.

## Setup Guide  

1. Install the `msgpack` Python package manually or via your preferred package manager.  
2. Clone the repo and copy `local_config.py.example` to `local_config.py`.  
3. Edit `local_config.py` to suit your network and hardware configuration.  
4. Run the following tests to ensure proper setup:  
   - `python test_server.py`: Verifies server functionality.  
   - `python test_noise.py`: Generates simple pulses for verification on an oscilloscope.  

## File Descriptions  

### Added  
- `experiment_multifreq.py`: New API for handling multi-frequency settings in MRI experiments.
### Modified
- `grad_board.py`: Includes improved DAC initialization sequence to avoid transient output errors.


### Original Files  
- `csvs/`: CSV files for `test_flocra_model.py`.  
- `experiment.py`: Basic API for controlling the MaRCoS server.  
- `examples.py`: Examples demonstrating the usage of `experiment.py` and other libraries [WIP].  
- `local_config.py.example`: Template for local configuration; copy and rename to `local_config.py` for customization.  
- `server_comms.py`: Low-level communication library for the MaRCoS server; use this to develop custom APIs.  
- `test_flocra.py`: Low-level examples/tests for the Flocra system.  
- `test_flocra_model.py`: Unit tests for the MaRCoS server and Verilator model of the Flocra HDL.  
- `test_ocra_pulseq.py`: Tests for the [ocra-pulseq](https://github.com/lcbMGH/ocra-pulseq) interface [WIP].  
- `test_server.py`: Unit tests for standalone MaRCoS server operations.  
- `visualiser.py`: [WIP] Basic visualizations of expected OCRA assembly file outputs.  

## Documentation  

For detailed usage instructions and internal workings, refer to the [MaRCoS wiki](https://github.com/vnegnev/marcos_extras/wiki):  
- [Using MaRCoS](https://github.com/vnegnev/marcos_extras/wiki/using_marcos)  
- [MaRCoS Internals](https://github.com/vnegnev/marcos_extras/wiki/marcos_internals)  

## Notes  

Some descriptions in this file may need updates as the project evolves. Contributions and suggestions are welcome!
