import DOMPurify from 'dompurify'

/**
 * Sanitize untrusted HTML content to prevent XSS attacks
 */
export function sanitizeHTML(dirty: string): string {
  // Configure DOMPurify - allow safe HTML for markdown rendering
  const config = {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'span',
      'div', 'a', 'img', 'button' // for markdown processing
    ],
    ALLOWED_ATTR: ['href', 'class', 'style', 'src'], // href for links, class/style for styling
    ALLOW_DATA_ATTR: false, // No data attributes
    FORBID_TAGS: ['script', 'object', 'embed', 'form', 'input', 'textarea'],
    FORBID_ATTR: ['onclick', 'onload', 'onerror', 'onmouseover', 'onmouseout', 'onkeydown', 'onkeyup', 'onkeypress']
  }

  return DOMPurify.sanitize(dirty, config)
}

/**
 * Sanitize user input for safe processing
 * Removes potentially dangerous characters and patterns
 */
export function sanitizeUserInput(input: string): string {
  if (!input || typeof input !== 'string') return ''

  return input
    // Remove null bytes and dangerous control characters
    .replace(/\0/g, '')
    // Remove control characters that aren't tabs/newlines
    .replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, '')
    // Remove HTML tags entirely (for basic input sanitization)
    .replace(/<[^>]*>/g, '')
    // Trim leading/trailing whitespace only
    .trim()
}

/**
 * Sanitize user input for real-time typing (preserves all normal typing)
 * Only removes null bytes for safety
 */
export function sanitizeUserInputTyping(input: string): string {
  if (!input || typeof input !== 'string') return ''

  return input
    // Only remove null bytes (most dangerous character)
    .replace(/\0/g, '')
}

/**
 * Sanitize system prompts to prevent injection attacks
 * This is critical for LLM security
 */
export function sanitizeSystemPrompt(prompt: string): string {
  if (!prompt || typeof prompt !== 'string') return ''

  return prompt
    // Remove potential injection markers
    .replace(/\{\{\s*(END_OF_SYSTEM_PROMPT|SYSTEM:|USER:)\s*\}\}/gi, '')
    // Remove instruction override attempts
    .replace(/^(Ignore|Forget|Disregard|Override).*/gim, '')
    // Remove comment injection attempts
    .replace(/<!--[\s\S]*?-->|\/\*[\s\S]*?\*\//g, '')
    // Limit maximum length
    .substring(0, 10000)
    .trim()
}

/**
 * Sanitize message content before rendering
 * Handles both regular text and HTML-generated content
 */
export function sanitizeMessageContent(content: string): string {
  // First sanitize as basic input
  const sanitizedText = sanitizeUserInput(content)

  // If it contains HTML-like structures, treat as potential HTML
  // but only allow safe markdown-generated HTML
  if (/<[a-z][\s\S]*>/i.test(sanitizedText)) {
    return sanitizeHTML(sanitizedText)
  }

  // Otherwise return as plain text (will be rendered via React)
  return sanitizedText
}

/**
 * Validate file upload types and sizes
 */
export function validateFileUpload(file: File): { valid: boolean; error?: string } {
  // Maximum file size: 10MB
  const maxSize = 10 * 1024 * 1024

  if (file.size > maxSize) {
    return { valid: false, error: 'File too large. Maximum size is 10MB.' }
  }

  // Allowed file types for text/PDF processing
  const allowedTypes = [
    'text/plain',
    'text/markdown',
    'text/csv',
    'application/pdf',
    'application/json',
    'text/javascript',
    'text/typescript',
    'text/python',
    'text/html',
    'text/css'
  ]

  // Check MIME type
  if (!allowedTypes.includes(file.type)) {
    // Fallback to file extension check for common file types
    const allowedExtensions = ['.txt', '.md', '.csv', '.pdf', '.json', '.js', '.ts', '.py', '.html', '.css']
    const fileName = file.name.toLowerCase()
    const hasAllowedExt = allowedExtensions.some(ext => fileName.endsWith(ext))

    if (!hasAllowedExt) {
      return {
        valid: false,
        error: 'Unsupported file type. Only text files, PDFs, and code files are supported.'
      }
    }
  }

  return { valid: true }
}
