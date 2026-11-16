import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
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

  it('生成结束但没有来源时不再显示光标/加载动画', () => {
    const message: Message = {
      id: '6',
      role: 'assistant',
      content: '未找到相关信息，请尝试换个问法。',
      sources: [],
      pending: false,
    };

    const { container } = render(<ChatMessage message={message} />);
    expect(screen.getByText('未找到相关信息，请尝试换个问法。')).toBeInTheDocument();
    expect(container.querySelector('.animate-bounce')).toBeNull();
    expect(container.querySelector('.animate-pulse')).toBeNull();
  });

  it('渲染可展开的引用片段', () => {
    const message: Message = {
      id: '7',
      role: 'assistant',
      content: '入职满三年后年假为 15 天。',
      sources: ['policy.txt'],
      citations: [
        { filename: 'policy.txt', text: '员工入职满三年后，年假增加到 15 天。' },
        { filename: 'policy.txt', text: '年假需要提前三个工作日提交申请。' },
      ],
      pending: false,
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText(/引用片段 · 2 段/)).toBeInTheDocument();
    expect(screen.getByText(/年假增加到 15 天/)).toBeInTheDocument();
    expect(screen.getByText(/提前三个工作日/)).toBeInTheDocument();
  });

  it('点击"复制回答"会写入剪贴板', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    const message: Message = {
      id: '8',
      role: 'assistant',
      content: '要复制的内容',
      pending: false,
    };

    render(<ChatMessage message={message} />);
    fireEvent.click(screen.getByText('复制回答'));
    expect(writeText).toHaveBeenCalledWith('要复制的内容');
    expect(await screen.findByText('已复制')).toBeInTheDocument();
  });

  it('停止生成后显示标记', () => {
    const message: Message = {
      id: '9',
      role: 'assistant',
      content: '只写了一半',
      pending: false,
      stopped: true,
    };

    render(<ChatMessage message={message} />);
    expect(screen.getByText('已停止生成')).toBeInTheDocument();
  });
});
