This repository contains sample code with results of 1:1 engagement with Action Press. Action Press LLC is a part of Action-Media company - the largest supplier of professional media in Russia. To inform customers about news and stimulate subscriptions Action Press has developed an internal outbound marketing system that regularly send emails to the customer base. System has used full-text search with a basic automation which was not so effective as expected, as users were required to manually check and sort replies on the emails, forwarding them to corresponding departments. Main types of consumers responses were autoreplies, requests to remove from the mailing list, clarifying questions and requests for new subscriptions. The core idea of the engagement with us was in development of a machine learning model to make the process as much automatic as possible and then operationalize it to be used with other parts of the system in production. In this repository you can find ML model building scripts and a way Azure Functions can be used as a host for python ML models in a simple and cost effective way.

Please, check the article for more info:
* [Russian]()

# Setup

## Development environment

Prerequisites:
Windows PC with the following software installed:
* Anaconda version 4 or higher. Download it [here](https://www.anaconda.com/download/) for free.
* Visual Studio Code with [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=donjayamanne.jupyter) extensions installed. Download it [here](https://code.visualstudio.com/) for free. Install extensions manually using the links or from VS Code itself.

_The code should possibly work on Linux and Mac, but hasn't been tested. A subsequent steps assumes that you have a PC with Windows 10._

1. Open the command line with current working directory set to any place you want this code to be. Clone the repository. Replace the url with your own if using the fork.

```
git clone https://github.com/evgri243/ap-sample.git
```
2. Change current working directory to the cloned folder.

```
cd ap-sample
```

3. Create Anaconda environment with all required packages installed using the command below. If not overwritten it will be called `ap-sample`.

```
conda env create -f environment.yml
```

4. Activate newly created environment.

```
activate ap-sample
```

5. Run Visual Studio Code from the same command prompt. This is required to correctly forward all environment settings associated with Python environment. Please, rerun steps 4-5 every time you want to work with this code.

```
code
```

6. Open _Folder_ with the whole repository in VS Code using GUI. And from Debug (_Ctrl+Shift+D_) run configuration named _Setup (Python)_ to install required NLTK corpora.

7. Replace `data\data.csv` with your own data in the same format and step through `Experiments\TrainingExperiment.py` to train your model.

8. Once ready, replace `Score\debug\input.csv` with your own data in the same format and step through `Score\Score.py` to score results.
 
Please, check the article for more info.

## Azure Deployment

The next steps require active Microsoft Azure subscription. If you don't have one, [create one for free](https://azure.microsoft.com/en-us/free/).

1. Fork the repository to your GitHub account.

2. Go to [portal.azure.com](https://portal.azure.com/) and create Azure Function, Click big _+_ ont the left side, search for _Function App_ and click Create. This steps deploys Azure Functions and Azure Storage to you subscription. For more information check the [docs](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-first-azure-function).

3. Using [Azure Portal](https://portal.azure.com/) configure integration between Azure Functions and the forked repo. Go to your function -> _Platform Features_ -> _Deployment option_, click _Setup_ and finish the wizard. Wait for initial deployment to succeed. Check this [docs](https://docs.microsoft.com/en-us/azure/azure-functions/functions-continuous-deployment#set-up-continuous-deployment) for additional information.

4. Go to the corresponding storage account (created with Function) using [Azure Portal](https://portal.azure.com/) or [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/). Put sample input file `Score\debug\input.csv` (or your owned data if model is changed) to _input_ folder of _mail-classify_ container. Wait for results to appear in _output_ folder.

Please, check the article for more info.