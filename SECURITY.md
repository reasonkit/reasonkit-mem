# Security Policy

## Supported Versions

We adhere to Semantic Versioning 2.0.0. Security updates are provided for the current major version.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in ReasonKit Memory, please report it privately:

1. **Email:** security@reasonkit.sh
2. **Response Time:** We are committed to responding to security reports within 48 hours.
3. **Process:**
   - We will investigate and verify the issue.
   - We will develop a patch.
   - We will release a security advisory and a patched version.
   - We will acknowledge your contribution (with permission).

## Responsible Disclosure

We ask that you:

- Give us reasonable time to fix the issue before making it public.
- Do not exploit the vulnerability to view data, modify data, or disrupt service.
- Do not attack our users or infrastructure.

## Security Considerations for Memory Infrastructure

### Data Storage

- **Embedded Mode (Qdrant):** Data is stored locally in the `./qdrant_data` directory by default. Ensure appropriate filesystem permissions.
- **Remote Qdrant:** When connecting to remote Qdrant instances, use TLS connections and API keys. Never commit credentials to version control.
- **Tantivy Index:** Full-text search indices are stored locally. Protect index directories with appropriate permissions.

### Embedding Providers

- **API Keys:** Store embedding provider API keys (OpenAI, Voyage, etc.) in environment variables, never in code.
- **Local Embeddings:** When using local embeddings (BGE-M3), model files are downloaded to local cache. Verify model checksums.
- **Data Transmission:** Embeddings sent to remote providers may contain sensitive information. Review your provider's data handling policies.

### Network Security

- **TLS:** All remote connections use TLS by default via the `reqwest` crate with `rustls`.
- **Timeouts:** Connection timeouts are enforced to prevent resource exhaustion.
- **Input Validation:** Query inputs are validated before processing to prevent injection attacks.

### Memory Safety

- This crate uses `#![forbid(unsafe_code)]` - no unsafe Rust code is present.
- All dependencies are audited via `cargo-audit` in CI.

## Security Audit

This project has undergone internal security audits. However, users should conduct their own security assessment before deploying in sensitive environments.

## Dependency Security

We use `cargo-deny` to ensure:

- No dependencies with known vulnerabilities (RUSTSEC advisories)
- No GPL-licensed dependencies (Apache 2.0 compatibility)
- No yanked crate versions
- Pinned dependency versions via `Cargo.lock`

## Best Practices for Users

1. **Isolate sensitive data:** Use separate vector collections for different data sensitivity levels.
2. **Encrypt at rest:** If storing sensitive embeddings, enable encryption on the storage layer.
3. **Audit access:** Log all access to retrieval endpoints in production.
4. **Regular updates:** Keep reasonkit-mem updated to receive security patches.
5. **Review embeddings:** Be aware that text embeddings can leak information about source content.
