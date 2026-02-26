# MEng

Update

Updating to see if this works after the reinstall 

another update

Will clean this up a little later but for now some details
I firtst set the remote to be gitlab this is where it pulls from
Then I added a second push destination
```
git remote set-url --push origin https://repo.ee.up.ac.za/isgpghg/mills_ds.git
git remote set-url --add --push origin git@github.com:Dean-Mills/meng.git
git remote -v
```

You should then see something like this

```
origin	https://repo.ee.up.ac.za/isgpghg/mills_ds.git (fetch)
origin	https://repo.ee.up.ac.za/isgpghg/mills_ds.git (push)
origin	git@github.com:Dean-Mills/meng.git (push)
```

Using Ubuntu 24.04 (so I have python installed check if you do)  

```
python3 --version
```

Then install latex (it's a bit excessive to do the full install but don't know what I will need so it is what it is) 

NOTE: If the install gets stuck just spam enter
```
sudo apt install texlive-full
```

# Setup 

I think for now the most logical thing is to create a single .venv in the root directory

There are a number of ways to do this

- Using visual studio code
    - ctrl-shift-p
    - Python: Create Environment...
- Using virtualenv
```
pip install virtualenv
virtualenv .venv
```
- Using python itself
```
python -m venv .venv
```

Activate the virtual .venv (this will be done automatically in vscode but if you used the other methods)

```
source .venv/bin/activate
```

Then install the requirements

```
pip install -r requirements.txt
```

To get matplotlib images I use TkAgg you might get a warning saying

```
Error loading image 000000363072.jpg: No module named 'tkinter'
```

The solution to this is

```
sudo apt-get install python3-tk
```
