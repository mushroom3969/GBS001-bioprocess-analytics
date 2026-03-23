"""Tab 7: Literature Analysis — PubMed search + Gemini synthesis."""
import streamlit as st
import time
import os
import re

from utils import (
    pubmed_search, pubmed_fetch_abstracts,
    build_pubmed_queries_with_gemini, call_gemini,
)


def _get_api_key():
    return st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))


def render():
    st.header("📚 文獻佐證分析")
    st.markdown("""
    <div class="info-box">
    自動從 <b>PubMed</b> 搜尋真實論文 → 抓取摘要 → Gemini 基於真實文獻整理分析。
    所有引用都附 PMID 連結，可直接追蹤原始論文。
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1：設定 ─────────────────────────────────────────
    st.markdown("### Step 1：設定分析參數")
    col_la, col_lb = st.columns(2)

    with col_la:
        target_var = st.text_input(
            "🎯 目標變數（Y）",
            value=st.session_state.get("target_col", ""),
            placeholder="例：phenyl chromatography_Yield Rate (%)")
        process_ctx = st.text_input(
            "🧪 製程背景",
            value="rhG-CSF protein purification, Phenyl Hydrophobic Interaction Chromatography")
        max_papers = st.slider("每個參數搜尋論文數", 1, 5, 3, key="lit_max_papers")

    with col_lb:
        auto_features = []
        if st.session_state.get("fi_perm_df") is not None:
            auto_features = st.session_state["fi_perm_df"]["Feature"].head(8).tolist()
        if st.session_state.get("pls_vip_df") is not None:
            vip_top = st.session_state["pls_vip_df"][
                st.session_state["pls_vip_df"]["VIP"] >= 1.0]["Feature"].head(8).tolist()
            auto_features = list(dict.fromkeys(auto_features + vip_top))

        feat_text = st.text_area(
            "📌 重要參數（每行一個，自動帶入分析結果）",
            value="\n".join(auto_features[:6]) if auto_features else "",
            height=200)
        lang = st.radio("輸出語言", ["繁體中文", "English"], horizontal=True, key="lit_lang")

    important_features = [f.strip() for f in feat_text.strip().split("\n") if f.strip()]

    if not important_features or not target_var.strip():
        st.warning("請填入目標變數與重要參數。")
        return

    # ── Step 2：PubMed 搜尋 ───────────────────────────────────
    st.markdown("### Step 2：搜尋 PubMed 文獻")

    if st.button("🔎 搜尋 PubMed 論文", key="run_pubmed"):
        api_key = _get_api_key()
        if not api_key:
            st.error("找不到 GEMINI_API_KEY。")
            return

        with st.spinner("🤖 Gemini 正在轉換搜尋關鍵字..."):
            queries = build_pubmed_queries_with_gemini(
                important_features, target_var, process_ctx, api_key)

        st.markdown("**生成的搜尋關鍵字：**")
        for feat, q in queries:
            st.caption(f"· `{feat[:50]}` → **{q}**")

        all_articles = {}
        prog = st.progress(0, text="搜尋中...")
        for i, (feat, query) in enumerate(queries):
            pmids = pubmed_search(query, max_results=max_papers)
            if pmids:
                arts = pubmed_fetch_abstracts(pmids)
                all_articles[feat] = {"query": query, "articles": arts}
            time.sleep(0.35)
            prog.progress((i + 1) / len(queries), text=f"搜尋中：{feat[:40]}...")
        prog.empty()

        st.session_state["pubmed_results"] = all_articles
        total = sum(len(v["articles"]) for v in all_articles.values())
        st.success(f"✅ 共找到 {total} 篇論文！")

    # ── 顯示論文 ─────────────────────────────────────────────
    if st.session_state.get("pubmed_results"):
        all_articles = st.session_state["pubmed_results"]
        st.markdown("#### 📄 找到的論文")
        for feat, data in all_articles.items():
            with st.expander(f"**{feat}** — {len(data['articles'])} 篇 | 搜尋：`{data['query']}`"):
                for art in data["articles"]:
                    st.markdown(
                        f"**[{art['title']}]({art['url']})**  \n"
                        f"_{art['journal']}_ ({art['year']}) · PMID: `{art['pmid']}`  \n"
                        f"{art['abstract']}..."
                    )
                    st.markdown("---")

        # ── Step 3：Gemini 分析 ──────────────────────────────
        st.markdown("### Step 3：AI 基於真實文獻分析")

        if st.button("🧠 開始 AI 文獻分析", type="primary", key="run_lit_gemini"):
            api_key = _get_api_key()
            if not api_key:
                st.error("找不到 GEMINI_API_KEY。")
                return

            # Build literature context
            lit_context = ""
            ref_list    = []
            ref_idx     = 1
            for feat, data in all_articles.items():
                lit_context += f"\n\n=== Parameter: {feat} ===\n"
                for art in data["articles"]:
                    lit_context += (
                        f"[{ref_idx}] {art['title']} "
                        f"({art['journal']}, {art['year']}, PMID:{art['pmid']})\n"
                        f"Abstract: {art['abstract']}\n\n"
                    )
                    ref_list.append(
                        f"[{ref_idx}] {art['title']}. "
                        f"{art['journal']} ({art['year']}). "
                        f"PMID: {art['pmid']}. {art['url']}"
                    )
                    ref_idx += 1

            lang_inst  = "Respond in Traditional Chinese (繁體中文)." if lang == "繁體中文" else "Respond in English."
            feat_list  = "\n".join([f"  {i+1}. {f}" for i, f in enumerate(important_features)])

            prompt = (
                "You are an expert bioprocess scientist. Analyze the following real PubMed literature "
                "to explain why the listed process parameters are important predictors of the target variable.\n\n"
                f"Process Context: {process_ctx}\n"
                f"Target Variable: {target_var}\n\n"
                f"Important Parameters:\n{feat_list}\n\n"
                f"=== REAL PUBMED LITERATURE ===\n{lit_context}\n\n"
                "STRICT RULES:\n"
                "1. ONLY cite papers from the list above using [number]\n"
                "2. Do NOT invent papers\n"
                "3. If evidence is weak or absent, say so\n"
                f"4. End with a Reference List with PMID and URL\n"
                f"5. {lang_inst}\n\n"
                "Output:\n"
                "## 總覽\n(collective effect on target)\n\n"
                "## 各參數分析\n(each param: mechanism + [ref])\n\n"
                "## 研究缺口\n(what is NOT covered by found literature)\n\n"
                "## 參考文獻\n(all cited with PMID + URL)"
            )

            with st.spinner("🧠 Gemini 正在基於真實文獻分析（約 30-60 秒）..."):
                try:
                    ai_response = call_gemini(api_key, prompt)
                    st.session_state.update({
                        "lit_response": ai_response,
                        "lit_ref_list": ref_list,
                        "lit_params": {
                            "target": target_var,
                            "features": important_features,
                            "context": process_ctx,
                        }
                    })
                    st.success("✅ 分析完成！")
                except Exception as e:
                    import traceback
                    st.error(f"Gemini 呼叫失敗：{e}")
                    if hasattr(e, "read"):
                        try:
                            st.code(e.read().decode("utf-8"), language="json")
                        except Exception:
                            pass
                    st.code(traceback.format_exc())

    # ── 顯示分析結果 ──────────────────────────────────────────
    if st.session_state.get("lit_response"):
        params   = st.session_state.get("lit_params", {})
        ref_list = st.session_state.get("lit_ref_list", [])

        st.markdown("---")
        st.markdown("### 📄 分析結果")
        st.caption(
            f"目標：`{params.get('target','')}` ｜ "
            f"參數數：{len(params.get('features',[]))} ｜ "
            f"引用論文：{len(ref_list)} 篇"
        )
        st.markdown(st.session_state["lit_response"])

        if ref_list:
            st.markdown("### 🔗 快速論文連結")
            for ref in ref_list:
                url_m = re.search(r"https://pubmed[^\s]+", ref)
                if url_m:
                    label = ref.split(url_m.group())[0].rstrip(". ")
                    st.markdown(f"- {label} [🔗 PubMed]({url_m.group()})")

        # Download
        st.markdown("---")
        export_md = (
            "# 文獻佐證分析報告\n"
            f"生成時間：{time.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## 製程背景\n{params.get('context','')}\n\n"
            f"## 目標變數\n{params.get('target','')}\n\n"
            "## 分析參數\n" + "\n".join(["- " + f for f in params.get("features", [])]) + "\n\n"
            "---\n\n## AI 分析結果（基於 PubMed 真實文獻）\n\n"
            + st.session_state["lit_response"] + "\n\n"
            "---\n\n## 所有搜尋到的論文\n\n" + "\n".join(ref_list)
        )
        st.download_button(
            "📥 下載完整報告（含文獻出處）",
            data=export_md.encode("utf-8"),
            file_name="pubmed_literature_report.md",
            mime="text/markdown",
            key="download_lit"
        )

        # Follow-up
        st.markdown("### 💬 追問")
        follow_up = st.text_area(
            "針對以上分析，想進一步了解？",
            placeholder="例：Loading capacity 對 yield 的非線性效應，文獻中有哪些實驗數據？",
            key="lit_followup_q")
        if st.button("📨 送出追問", key="lit_followup_btn") and follow_up.strip():
            api_key = _get_api_key()
            lang_fu = "Respond in Traditional Chinese." if lang == "繁體中文" else "Respond in English."
            with st.spinner("思考中..."):
                try:
                    reply = call_gemini(api_key, (
                        "Based on this previous analysis:\n\n"
                        + st.session_state["lit_response"]
                        + f"\n\nUser follow-up: {follow_up}\n\n"
                        "Answer using ONLY the cited literature. "
                        f"If more papers are needed, say so clearly.\n{lang_fu}"
                    ), max_tokens=2000)
                    st.markdown("#### 💬 回覆")
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"追問失敗：{e}")
