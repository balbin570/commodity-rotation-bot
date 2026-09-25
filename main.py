
# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse,
)
def dashboard():

    html = """

    <!DOCTYPE html>

    <html>

    <head>

    <meta charset="UTF-8">

    <title>
    Commodity Rotation Bot
    </title>

    <style>

    body {

        font-family:
        Arial,
        sans-serif;

        max-width:
        1100px;

        margin:
        40px auto;

        padding:
        20px;

        background:
        #f4f4f4;
    }

    .card {

        background:
        white;

        padding:
        20px;

        margin-bottom:
        20px;

        border-radius:
        10px;
    }

    a {

        display:
        block;

        margin:
        8px 0;

        font-size:
        17px;
    }

    </style>

    </head>

    <body>

    <h1>
    Commodity Rotation Bot v1.2A
    </h1>

    <p>
    Long-only commodity ETF/ETP scanner
    </p>

    <div class="card">

    <h2>
    METALLER
    </h2>

    <a href="/analyze/GLD">
    Altın - GLD
    </a>

    <a href="/analyze/SLV">
    Gümüş - SLV
    </a>

    <a href="/analyze/CPER">
    Bakır - CPER
    </a>

    </div>


    <div class="card">

    <h2>
    ENERJİ
    </h2>

    <a href="/analyze/USO">
    Petrol - USO
    </a>

    <a href="/analyze/UNG">
    Doğal Gaz - UNG
    </a>

    </div>


    <div class="card">

    <h2>
    TARIM
    </h2>

    <a href="/analyze/DBA">
    Tarım Sepeti - DBA
    </a>

    <a href="/analyze/WEAT">
    Buğday - WEAT
    </a>

    <a href="/analyze/CORN">
    Mısır - CORN
    </a>

    <a href="/analyze/SOYB">
    Soya - SOYB
    </a>

    </div>


    <div class="card">

    <h2>
    GENEL
    </h2>

    <a href="/analyze/DBC">
    Geniş Emtia Sepeti - DBC
    </a>
