# Inwards Re

In the InwardsRe package, there is a Jupyter Notebook for each task that has been automated.

## Requirements

To run the Jupyter Notebooks, you will need to have installed the following programs:

- [Visual Studio Code](https://code.visualstudio.com/)
- [GitHub Desktop](https://desktop.github.com/)
- Python (only through IT portal via request)

> **Note:** for the Python install, make sure to mention to ITS that you **need** Python installed on your physical machine, not in Citrix. If they install in both, that's fine, but the priority is to have it in your physical devide.

### After the above is installed

After installing the programs above, clone repository with GitHub Desktop. If you don't know how to do that, [follow this tutorial](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop)

Then open Command Prompt and run the following command:

```
pip install -r requirements.txt
```

This command install all the necessary library in order to run any code in this package.

For this command to work, you will need to `cd` to the correct folder where the requirements.txt is located.

If you need a guide to change folder in the Command Prompt, [follow this link](https://www.howtogeek.com/659411/how-to-change-directories-in-command-prompt-on-windows-10/).

If you prefer, you can also use the full filepath to `requirements.txt` file. For example, if you put the text file in your Desktop, you'd run the command like this:

```
pip install -r C:\Users\<your-user-name>\Desktop\requirements.txt
```
