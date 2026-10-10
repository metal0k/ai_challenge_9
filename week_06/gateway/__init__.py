"""Private LLM gateway: auth, limits and on-demand lifecycle in front of llama-server.

Self-contained on purpose: deploy copies only this package, so it imports nothing from
the rest of the project.
"""
