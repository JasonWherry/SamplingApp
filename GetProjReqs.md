### How to add requirements for specified project (Bash & pip3)
___

In console, navigate to your project folder.

Type the following in the console and hit **'enter'**.
```bash
pip3 install pipreqs
```

Type the following in the console and hit **'enter'**.
```bash
pwd
```
Copy the output from the console. It should look something like &rarr; **PATH/TO/PROJ/FOLDER**.
&nbsp;

Type the following in the console. Replace **PATH/TO/PROJ/FOLDER** by pasting your path, then hit **'enter'**.

```bash
pipreqs /PATH/THAT/YOU/COPIED --force
```

A **requirements.txt** file should appear with a list of project dependencies.