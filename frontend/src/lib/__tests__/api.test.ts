import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as api from '../api';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('API', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe('getStats', () => {
    it('returns stats data', async () => {
      const mockData = {
        document_count: 2,
        total_chars: 1000,
        documents: ['doc1.txt', 'doc2.pdf'],
        chunk_count: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData),
      });

      const result = await api.getStats();
      expect(result).toEqual(mockData);
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/stats',
        expect.objectContaining({ signal: expect.any(AbortSignal) })
      );
    });
  });

  describe('uploadDocument', () => {
    it('uploads file and returns response', async () => {
      const mockResponse = {
        message: '上传成功',
        filename: 'test.txt',
        chars: 100,
        chunks: 5,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const result = await api.uploadDocument(file);

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/upload',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData),
        })
      );
    });

    it('throws error on failed upload', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: '文件过大' }),
      });

      const file = new File(['test'], 'test.txt');
      await expect(api.uploadDocument(file)).rejects.toThrow('文件过大');
    });
  });

  describe('deleteDocument', () => {
    it('sends delete request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await api.deleteDocument('test.txt');
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/documents/test.txt',
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  describe('query', () => {
    it('sends question and returns answer', async () => {
      const mockResponse = {
        answer: '人工智能是...',
        sources: ['doc.txt'],
        conversation_id: 'conv-123',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.query('什么是AI？');
      expect(result).toEqual(mockResponse);
    });
  });
});
