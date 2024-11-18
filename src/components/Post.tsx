// src/components/Post.tsx
import React from 'react';
import { useParams } from 'react-router-dom';
import MarkdownRenderer from './MarkdownRenderer';
import { Box, Heading } from '@chakra-ui/react';

const markdownFiles = import.meta.glob('../posts/*.md', { as: 'raw' });

const Post: React.FC = () => {
  const { postName } = useParams<{ postName: string }>();
  const [content, setContent] = React.useState<string>('');
  const [error, setError] = React.useState<string>('');

  React.useEffect(() => {
    const loadMarkdown = async () => {
      try {
        const filePath = `../posts/${postName}.md`;
        if (markdownFiles[filePath]) {
          const markdown = await markdownFiles[filePath]();
          setContent(markdown);
        } else {
          throw new Error('Markdown file not found');
        }
      } catch (err) {
        console.error(err);
        setError('Post not found.');
      }
    };

    if (postName) {
      loadMarkdown();
    }
  }, [postName]);

  if (error) {
    return (
      <Box p={4}>
        <Heading>{error}</Heading>
      </Box>
    );
  }

  if (!content) {
    return (
      <Box p={4}>
        <Heading>Loading...</Heading>
      </Box>
    );
  }

  return (
    <Box p={4} dir="ltr" textAlign="left">
      <Heading as="h1" size="xl" mb={4}>
        {postName.replace(/-/g, ' ')}
      </Heading>
      <MarkdownRenderer content={content}
      meta={{
            title: "My Article",
            description: "Article description",
            image: "https://example.com/image.jpg",
            url: "https://example.com/article"
        }}
        />
    </Box>
  );
};

export default Post;
