from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text(encoding='utf-8') if (HERE / 'README.md').exists() else ''

# Read requirements.txt if present
req_file = HERE / 'requirements.txt'
if req_file.exists():
    install_requires = [l.strip() for l in req_file.read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
else:
    install_requires = []

setup(
    name='SummarizerBot',
    version='0.1.0',
    description='Discord bot that summarizes conversations using LLMs',
    long_description=README,
    long_description_content_type='text/markdown',
    author='ElectrodeYT',
    packages=find_packages(exclude=('tests', 'docs')),
    py_modules=['main', 'cache', 'summary_llm'],
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'summarizerbot = main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
