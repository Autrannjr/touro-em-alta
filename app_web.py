"""Touro em Alta - Screener de Abertura (versao web/celular, Streamlit)."""
import hmac
import io

import pandas as pd
import streamlit as st

import engine

st.set_page_config(page_title="Touro em Alta", page_icon="icone.png", layout="centered")

DISCLAIMER = (
    "Ferramenta de triagem, não é recomendação de investimento. Não prevê o mercado e não garante "
    "resultado. Day trade pode causar perdas. Você decide e executa as ordens na sua corretora."
)

st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-bottom:2rem}"
    "div[data-testid='stMetricValue']{font-size:1.15rem}</style>",
    unsafe_allow_html=True,
)


def check_password() -> bool:
    """Se existir APP_PASSWORD nos secrets, exige senha. Sem secret, acesso livre."""
    try:
        senha = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        senha = ""
    if not senha:
        return True
    if st.session_state.get("ok"):
        return True
    st.title("🐂 Touro em Alta")
    typed = st.text_input("Senha", type="password")
    if typed:
        if hmac.compare_digest(typed, senha):
            st.session_state["ok"] = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    return False


@st.cache_data(ttl=300, show_spinner=False)
def scan(cap, risk, top, bdr, sl, tp):
    df, info = engine.run_scan(cap, risk, top, bdr, sl, tp)
    news = {r["ticker"]: engine.headlines(r["ticker"], r["tipo"], r["subjacente"]) for _, r in df.iterrows()}
    return df, info, news


def card(idx, r, news):
    gap = "-" if r["gap"] is None or r["gap"] != r["gap"] else f"{r['gap'] * 100:+.1f}%"
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.subheader(f"{idx}. {r['ticker']}")
        h1.caption(r["tipo"] + (f" · mãe {r['subjacente']}" if r["subjacente"] else ""))
        h2.metric("Score", f"{r['score']:.0f}")
        a, b, c = st.columns(3)
        a.metric("Entrada", f"{r['entry']:.2f}")
        b.metric("Stop", f"{r['stop']:.2f}")
        c.metric("Alvo", f"{r['target']:.2f}")
        d, e, f = st.columns(3)
        d.metric("Qtd", f"{r['qty']}")
        e.metric("Risco R$", f"{r['risk_brl']:.0f}")
        f.metric("Alvo/Risco", f"{r['rr']:.2f}")
        st.caption(
            f"Preço {r['price']:.2f} · Gap {gap} · Vol.rel {r['rvol']:.1f}x · ATR {r['atr_pct'] * 100:.1f}% · "
            f"RSI {r['rsi']:.0f}"
        )
        with st.expander("Detalhes e motivos"):
            if r["tipo"] == "BDR" and r["subjacente"]:
                ur = r["under_ret1"]
                ur_txt = f"{ur * 100:+.1f}%" if ur is not None and ur == ur else "n/d"
                st.write(
                    f"Ação-mãe {r['subjacente']} (último pregão dos EUA: {ur_txt}). "
                    "O BDR só acompanha bem a ação americana com a bolsa dos EUA aberta (11h30 em Brasília)."
                )
            st.markdown("**Por que está no topo**")
            for w in r["why"] or ["pontuação equilibrada nos critérios"]:
                st.write(f"- {w}")
            st.write(
                f"Variação 5 dias {r['ret5'] * 100:+.1f}% · liquidez média R$ {r['avg_fin'] / 1e6:.1f} mi/dia. "
                "Abaixo de 100 ações é mercado fracionário (ticker com F). "
                "Se bater stop ou alvo, encerre; se não, zere antes do fechamento."
            )
            st.markdown("**Manchetes recentes**")
            for h in news.get(r["ticker"], []) or ["(nenhuma encontrada)"]:
                st.write(f"- {h}")


def main():
    if not check_password():
        return
    st.title("🐂 Touro em Alta")
    st.caption("Screener de abertura · B3 + BDRs")

    with st.expander("⚙️ Configurações", expanded=False):
        c1, c2 = st.columns(2)
        cap = c1.number_input("Capital (R$)", min_value=100.0, value=10000.0, step=500.0)
        risk = c2.number_input("Risco por operação (%)", min_value=0.1, value=1.0, step=0.1)
        c3, c4 = st.columns(2)
        sl = c3.number_input("Stop (xATR)", min_value=0.1, value=0.6, step=0.1)
        tp = c4.number_input("Alvo (xATR)", min_value=0.1, value=0.9, step=0.1)
        top = st.slider("Nº de ativos", 1, 10, 5)
        bdr = st.checkbox("Incluir BDRs", value=True)

    if st.button("Analisar agora", type="primary", use_container_width=True):
        with st.spinner("Baixando dados e analisando (1 a 2 minutos)..."):
            try:
                st.session_state["res"] = scan(cap, risk, top, bdr, sl, tp)
            except Exception as e:  # noqa: BLE001
                st.error(f"Erro: {e}")
                st.session_state.pop("res", None)

    res = st.session_state.get("res")
    if res:
        df, info, news = res
        if df.empty:
            st.warning("Nenhum ativo passou nos filtros (dados indisponíveis ou fora do horário). Tente de novo em instantes.")
        else:
            modo = "pregão em andamento" if info["has_today"] else "pré-abertura/fechado (dados do último fechamento)"
            st.info(
                f"{info['now']:%d/%m %H:%M} · {modo} · {info['scanned']} analisados · "
                f"{len(info['missing'])} sem dados · {len(info['illiquid'])} pouca liquidez"
            )
            for idx, r in df.iterrows():
                card(idx, r, news)
            buf = io.BytesIO()
            df.drop(columns=["why", "has_today"]).to_excel(buf, index_label="#")
            st.download_button(
                "Baixar Excel", buf.getvalue(), "screener_abertura.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True,
            )
    st.caption(DISCLAIMER)


main()
