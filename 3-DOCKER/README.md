# Local Python Inference Test


```bash
cd 3-serve
cd opt_program
python run_inference.py
```


# Docker Build and Test


## Directory


```bash
cd 3-serve
```



## Local Test


### build and docker

```bash
bash build/build_serve_local.sh "serve-docker-demo" 
```

### setup server

```bash
bash test/serve_local.sh "serve-docker-demo" 
```




## ECR Test


### build docker for ecr

You need to copy your AWS tokens. 


```bash
bash build/build_serve_ecr.sh "pipeline-server-docker-v1103"
```

### setup server

```bash
bash test/serve_local.sh  "pipeline-server-docker-v1103"
```

