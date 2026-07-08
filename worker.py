import traceback

try:
    import runpod
    from handler import handler

    print("Worker starting...")
    runpod.serverless.start({
        "handler": handler
    })

except Exception as e:
    print("WORKER STARTUP ERROR:")
    print(str(e))
    traceback.print_exc()
    raise