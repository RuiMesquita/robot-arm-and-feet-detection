## Feet Detection using u-net segmentation

## Get started
### Activate environment:

List available environments with:
```bash
conda info --envs
```

Create environment with:
```bash
conda env create -prefix lib/<env-name> --file lib/environment.yaml
```

Activate environment with:
```bash
conda activate <environment>
```

To add more dependencies to the project edit the environment.yaml file in lib directory and update the environment with the following command:
```bash
conda env update -prefix lib/<env-name> --file lib/environment.yaml --prune
```

Deactivate the current environment with:
```bash
conda deactivate
```

### Run unittest

Inside tests folder you will find multiple classes with unittests for several modules on the porject. In order to run all tests run:
```bash
python -m unittest -v unit_test.py
``` 
