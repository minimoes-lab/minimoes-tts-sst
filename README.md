---
title: Fastapi
emoji: 😻
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
---
CMD ["uvicorn","api:app","--workers","1","--timeout","30000","--graceful-timeout","30000","--bind","0.0.0.0:7860","--worker-class","uvicorn.workers.UvicornWorker"]

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
