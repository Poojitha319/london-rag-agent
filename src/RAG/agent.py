class SimpleAgent:
    def __init__(self, rag):
        self.rag = rag

    def clarify(self, q: str) -> str:
        if "cheap" in q.lower():
            return q + " under £500000"
        return q

    def plan(self, clarified: str):
        plan = {"filters": {}, "tool": "filter"}
        q = clarified.lower()
        import re
        m = re.search(r"(\d+)\s*bed", q)
        if m:
            plan["filters"]["bedrooms"] = int(m.group(1))
        for b in ["camden", "westminster", "hackney", "lambeth", "greenwich", "islington"]:
            if b in q:
                plan["filters"]["borough"] = b.title()
        m2 = re.search(r"under\s*£?([0-9,]+)k?", q)
        if m2:
            val = int(m2.group(1).replace(",", ""))
            if "k" in q:
                val *= 1000
            plan["filters"]["max_price"] = val
        return plan

    def execute(self, plan):
        df = self.rag.df.copy()
        f = plan["filters"]
        if "borough" in f:
            df = df[df["borough"].str.contains(f["borough"], case=False, na=False)]
        if "bedrooms" in f:
            df = df[df["bedrooms"] == f["bedrooms"]]
        if "max_price" in f:
            df = df[df["price"] <= f["max_price"]]
        return df.sort_values("price").head(5).to_dict(orient="records")

    def respond(self, results):
        if not results:
            return {"answer": "No matching properties found.", "citations": []}
        lines, cites = [], []
        for r in results:
            lines.append(f"- {r['address']} (£{int(r['price'])}, ID: {r['property_id']})")
            cites.append(r["property_id"])
        return {"answer": "Found properties:\n" + "\n".join(lines), "citations": cites}

    def run(self, q: str):
        steps = []
        clarified = self.clarify(q)
        steps.append({"step": "clarify", "output": clarified})
        plan = self.plan(clarified)
        steps.append({"step": "plan", "output": plan})
        results = self.execute(plan)
        steps.append({"step": "execute", "rows": len(results)})
        final = self.respond(results)
        steps.append({"step": "respond", "output": final})
        return {"steps": steps, "final": final}
