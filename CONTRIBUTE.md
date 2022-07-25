# Contributing to GAN Facies Modeling

Everyone can contribute to GAN Facies Modeling, and we value everyone’s contributions. There are several ways to contribute, including:

- Raising [issue](https://github.com/valentingol/my_workflow_template/issues) on the Github repository

- Proposing [Pull requests](https://github.com/valentingol/my_workflow_template/pulls) to the Github repository

- Contact me by email (valentin.goldite@gmail.com)

- Create your own repository based on this one and cite it

## Pull request checklist

Before proposing a PR you must follow some rule:

- Code should work on Python 3.8-3.10

- Code should respect [PEP8](https://peps.python.org/pep-0008/)

- The format of the docstrings follows [Numpy guidline](https://numpydoc.readthedocs.io/en/latest/format.html)

**IMPORTANT: Before submitting a PR you should run the following command:**

```script
sh tests/basic_checks.sh
```

This script run all the unit tests and use the following linter: `mypy` (typing checks), `isort` (import order checks), `pydocstyle` (docstrings checks), `flake8` (PEP8 checks), `pylint` (style checks). If the unit tests pass (**try to keep the unit tests coverage high**), the linter checks pass and your pylint score is above the minimum score (that is 9.0), you can make the PR (**try to keep the pylint score as much high as possible!**).

## Commit messages

Commits should start with an emoji and directly followed by a descriptive and precise message that starts with a capital letter and should be written in present tense. E.g:

*✨: added configuration function* ❌ Bad

*✨ Add function to save configuration file* ✅ Good

Emojis not only look great but also makes you rethink what to add to a commit. The goal is to dedicate each single kind of change to a single commit. Make many but small commits!

Emojis of commit message follow mainly the [Gitmoji](https://gitmoji.dev/) guideline (the different ones start with an asterisk *). The most useful are:

| Emoji                                 | Description                                     |
| ------------------------------------- | ----------------------------------------------- |
| 🎉 `:tada:`                        | Initial commit                                  |
| ✨ `:sparkles:`                    | New cool feature                                |
| ➕ `:heavy_plus_sign:` *           | Add file and/or folder                          |
| 🔥 `:fire:`                        | Remove some code or file                        |
| 📝 `:memo:`                        | Add or improve readme, docstring or comments    |
| 🎨 `:art:`                         | Improve style, format or structure  of the code |
| ♻️ `:recycle:`                       | Refactor the code                               |
| 🚚 `:truck:`                       | Rename and/or move files and folders            |
| 🐛 `:bug:` OR 🪲 `:beetle:` *   | Fix a bug                                       |
| ✏️  `:pencil2:`                      | Fix typo                                        |
| 🔧 `:wrench:`                      | Add or update configuration files               |
| 🍱 `:bento:`                       | Add or update assets                            |
| 🚀 `:rocket:` *                    | Improve performance                             |
| ⚗️ `:alembic:`                       | Perform experiment                              |
| 🚸 `:children_crossing:`           | Improve user experience                         |
| 🆙 `:up:` * OR 🔖 `:bookmark:`  | Update the version/tag                          |
| ⬆️  `:arrow_up:`                     | Upgrade dependency                              |
| 🚧 `:construction:`                | Work in progress                                |
| 🔀 `:twisted_rightwards_arrows:`   | Merge a branch                                  |
| Check [Gitmoji](https://gitmoji.dev/) | *OTHER*                                         |

Installing the [Gitmoji VSCode extension](https://marketplace.visualstudio.com/items?itemName=seatonjiang.gitmoji-vscode) can be useful to get the emoji you want quickly.

## Version and tag numbers

Version/tag numbers will be assigned according to the [Semantic Versioning scheme](https://semver.org/). This means, given a version number MAJOR.MINOR.PATCH, we will increment the:

- MAJOR version when we make incompatible API changes
- MINOR version when we add functionality in a backwards compatible manner
- PATCH version when we make backwards compatible bug fixes
