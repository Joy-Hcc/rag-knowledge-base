import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ChatMessage, { type Message } from '../ChatMessage';

describe('ChatMessage', () => {
  it('renders user message', () => {
    const message: Message = {
      id: '1',
      role: 'user',
      content: '什么是人工智能？',
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText('什么是人工智能？')).toBeInTheDocument();
  });

  it('renders assistant message', () => {
    const message: Message = {
      id: '2',
      role: 'assistant',
      content: '人工智能是计算机科学的一个分支。',
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText('人工智能是计算机科学的一个分支。')).toBeInTheDocument();
  });

  it('renders assistant message with sources', () => {
    const message: Message = {
      id: '3',
      role: 'assistant',
      content: '根据文档...',
      sources: ['doc1.txt', 'doc2.pdf'],
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText('根据文档...')).toBeInTheDocument();
    expect(screen.getByText(/doc1\.txt/)).toBeInTheDocument();
    expect(screen.getByText(/doc2\.pdf/)).toBeInTheDocument();
  });

  it('renders loading state for empty assistant message', () => {
    const message: Message = {
      id: '4',
      role: 'assistant',
      content: '',
      sources: [],
    };

    const { container } = render(<ChatMessage message={message} />);
    // 检查是否有动画点（loading 指示器）
    expect(container.querySelector('.animate-bounce')).toBeInTheDocument();
  });

  it('renders error message', () => {
    const message: Message = {
      id: '5',
      role: 'assistant',
      content: '请求失败: 网络错误',
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText('请求失败: 网络错误')).toBeInTheDocument();
  });
});
