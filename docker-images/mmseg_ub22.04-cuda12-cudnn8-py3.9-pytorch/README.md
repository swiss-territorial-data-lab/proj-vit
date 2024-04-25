# ub22.04-py3.10

Image ubuntu 22.04 avec python 3.10 et pip

## Build

`
docker build -t ub22.04-py3.10 .
`

### Build with poetry dependency

1. Place the pyproject.toml of your project near the Dockerfile
2. Uncomment the 5 lines under `# Install dep with poetry` in the dockerfile
3. Run the docker build command

## Run

`
docker run -it --rm  --name my_docker  ub22.04-py3.10
`