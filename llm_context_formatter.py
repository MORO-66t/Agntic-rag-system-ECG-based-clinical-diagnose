class LLMContextFormatter:

    @staticmethod
    def build(ctx):

        lines = []

        lines.append(
            f"Detected Condition: {ctx['condition_id']}"
        )

        lines.append("")

        for chunk in ctx["knowledge"]:

            title = chunk["section"].replace(
                "_",
                " "
            ).title()

            lines.append(
                f"## {title}"
            )

            lines.append(
                chunk["content"]
            )

            lines.append("")

        return "\n".join(lines)