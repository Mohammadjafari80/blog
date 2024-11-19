// src/components/Post.tsx
import React from 'react';
import { useParams } from 'react-router-dom';
import MarkdownRenderer from './MarkdownRenderer';
import { Box, Heading, Skeleton, SkeletonText } from '@chakra-ui/react';

const markdownFiles = import.meta.glob('../posts/*.md', { as: 'raw' });

const Post: React.FC = () => {
  const { postName } = useParams<{ postName: string }>();
  const [content, setContent] = React.useState<string>('');
  const [meta, setMeta] = React.useState<Record<string, string>>({});
  const [error, setError] = React.useState<string>('');
  const [loading, setLoading] = React.useState<boolean>(true);

  React.useEffect(() => {
    const loadMarkdown = async () => {
      try {
        const filePath = `../posts/${postName}.md`;
        if (markdownFiles[filePath]) {
          const rawMarkdown = await markdownFiles[filePath]();

          // Extract metadata from the comment block
          const metaMatch = rawMarkdown.match(/<!--\s*({[\s\S]*?})\s*-->/);
          const metadata = metaMatch ? JSON.parse(metaMatch[1]) : {};

          // Remove metadata block from content
          const cleanedContent = rawMarkdown.replace(/<!--[\s\S]*?-->/, '').trim();

          setMeta(metadata);
          setContent(cleanedContent);
        } else {
          throw new Error('Markdown file not found');
        }
      } catch (err) {
        console.error(err);
        setError('Post not found.');
      } finally {
        setLoading(false);
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

  if (loading) {
    return (
      <Box p={4}>
        <Skeleton height="40px" mb={4} />
        <Skeleton height="600px" mb={4} />
        <SkeletonText noOfLines={7} spacing="4" />
      </Box>
    );
  }

  return (
    <Box p={4} dir="ltr" textAlign="left">
      <MarkdownRenderer content={content} meta={meta} />
    </Box>
  );
};

export default Post;
