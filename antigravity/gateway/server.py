@app.post("/api/account/connect")
async def connect_account(req: AccountConnectRequest):
    if not req.api_key or len(req.api_key) < 8:
        raise HTTPException(status_code=400, detail="Invalid API key.")

    if not hasattr(app_state, 'accounts'):
        app_state.accounts = {}
    if not hasattr(app_state, 'autotrade_engines'):
        app_state.autotrade_engines = {}

    if req.exchange == "cryptocom":
        try:
            client = CryptoComClient(req.api_key, req.api_secret)
            balances = await client.get_balance()
            log.info("account.verified", exchange="cryptocom", balances=balances)
        except Exception as exc:
            raise HTTPException(status_code=401, detail=f"Crypto.com verification failed: {exc}")

        if req.auto_trade:
            engine = AutotradeEngine(
                client=client,
                instrument="BTC_USDT",
                max_position_usdt=req.max_position_size * 50000,
            )
            app_state.autotrade_engines["cryptocom"] = engine
        else:
            app_state.autotrade_engines.pop("cryptocom", None)
    else:
        balances = {}

    app_state.accounts[req.exchange] = {
        "api_key": req.api_key[:8] + "••••••",
        "auto_trade": req.auto_trade,
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "success": True,
        "exchange": req.exchange,
        "auto_trade": req.auto_trade,
        "balances": balances,
        "message": f"Connected via crypto-com-app skill. Auto-trade {'enabled' if req.auto_trade else 'disabled'}.",
    }