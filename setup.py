
"""Setup JAX-CFD."""
import setuptools

base_requires = ['jax','jax_cfd','jax-md', 'numpy', 'scipy', 'tree-math']
data_requires = ['matplotlib', 'seaborn', 'Pillow', 'xarray']
ml_requires = ['dm-haiku', 'einops', 'gin-config']
tests_requires = ['absl-py', 'pytest', 'pytest-xdist', 'scikit-image']

setuptools.setup(
    name='jax_ib',
    version='0.1.0',
    license='Apache 2.0',
    author='Alhashim',
    author_email='mohammed.mga122@gmail.com',
    install_requires=base_requires,
    extras_require={
        'data': data_requires,
        'ml': ml_requires,
        'tests': tests_requires,
        'complete': data_requires + ml_requires + tests_requires,
    },
    url='https://github.com/hashimmg/jax_IB',
    packages=setuptools.find_packages(),
    python_requires='>=3.12',
)
