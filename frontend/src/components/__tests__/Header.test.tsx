import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Header from '../Header';

describe('Header', () => {
  it('renders title', () => {
    render(<Header />);
    expect(screen.getByText('AI 知识库')).toBeInTheDocument();
  });

  it('renders subtitle', () => {
    render(<Header />);
    expect(screen.getByText('上传文档，向 AI 提问')).toBeInTheDocument();
  });

  it('shows new conversation button when hasConversation is true', () => {
    const onNewConversation = vi.fn();
    render(
      <Header onNewConversation={onNewConversation} hasConversation={true} />
    );

    const button = screen.getByText('+ 新建对话');
    expect(button).toBeInTheDocument();

    fireEvent.click(button);
    expect(onNewConversation).toHaveBeenCalledOnce();
  });

  it('hides new conversation button when hasConversation is false', () => {
    render(<Header onNewConversation={vi.fn()} hasConversation={false} />);
    expect(screen.queryByText('+ 新建对话')).not.toBeInTheDocument();
  });

  it('hides new conversation button when no handler provided', () => {
    render(<Header hasConversation={true} />);
    expect(screen.queryByText('+ 新建对话')).not.toBeInTheDocument();
  });
});
