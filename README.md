# Multi-Car Racing with PyTorch

## Dependencies
- [pytorch 0.4](https://pytorch.org/)
- [gym 0.10](https://github.com/openai/gym)
- [visdom 0.1.8](https://github.com/facebookresearch/visdom)

## Training
Start a Visdom server with ```python -m visdom.server```. It will serve on http://localhost:8097/ by default.

To train the agent, run```python train.py --render --vis``` or ```python train.py --render``` (to train without Visdom). 
To test, run ```python test.py --render```.
