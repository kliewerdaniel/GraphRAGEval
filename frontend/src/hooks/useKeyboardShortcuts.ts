import { useEffect, useCallback } from 'react'

type KeyboardShortcut = {
  key: string
  ctrlKey?: boolean
  shiftKey?: boolean
  metaKey?: boolean
  altKey?: boolean
  action: () => void
  description: string
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  const handleKeyPress = useCallback((event: KeyboardEvent) => {
    const { key, ctrlKey, shiftKey, metaKey, altKey } = event

    for (const shortcut of shortcuts) {
      const matchesKey = shortcut.key.toLowerCase() === key.toLowerCase()
      const matchesCtrl = (shortcut.ctrlKey ?? false) === ctrlKey
      const matchesShift = (shortcut.shiftKey ?? false) === shiftKey
      const matchesMeta = (shortcut.metaKey ?? false) === metaKey
      const matchesAlt = (shortcut.altKey ?? false) === altKey

      if (matchesKey && matchesCtrl && matchesShift && matchesMeta && matchesAlt) {
        event.preventDefault()
        shortcut.action()
        break
      }
    }
  }, [shortcuts])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [handleKeyPress])
}

export type { KeyboardShortcut }
