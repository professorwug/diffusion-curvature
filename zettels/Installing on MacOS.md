To install the full version of our package, which includes competitors' curvature algorithms, Mac users need to do some extra work.

## NetworKit (Used to compute the Ollivier Ricci and Forman Ricci Curvatures)

1. Install Homebrew (brew.sh), if not already installed. 
2. Run `brew install llvm`, then symlink the compiler where it's accessible to pip with

```sh
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
```
For future installs, you'll want to add these to your `.zshrc` startup profile, so the environment variables are set on login.

3. Run `brew install libomp`

You're now good to go. Congrats.