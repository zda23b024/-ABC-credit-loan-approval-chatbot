# Vercel runtime limitation

The current Vercel Python Function deployment fails when LightGBM loads because
the runtime does not provide `libgomp.so.1`. The pinned `lightgbm==4.7.0` wheel
links to GNU OpenMP; installing the Python package alone does not install the
missing operating-system library. Vercel's documented Python Function
configuration can bundle project files, but does not provide a documented
system-package installation step for that runtime.

The project owner requested that Docker not be used. Under that constraint,
there is no supported way in the standard Vercel Python Function to add
`libgomp1` to the function's operating system. The chatbot's saved model and
frontend remain unchanged, but deploying its backend on Vercel's standard
Python runtime will continue to fail at LightGBM import.

Use the native Railpack setup in [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)
instead. It installs the distro package `libgomp1` as a runtime dependency and
serves the existing FastAPI app and frontend together. It uses no Dockerfile,
Docker CLI, or custom shared library binary. The GitHub Linux smoke workflow
also installs `libgomp1` on Ubuntu and verifies model loading and synthetic API
outcomes.

Sources: [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python),
[Railpack package installation](https://railpack.com/guides/installing-packages/),
[Railpack configuration](https://railpack.com/config/file/).
