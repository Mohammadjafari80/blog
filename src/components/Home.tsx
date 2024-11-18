// src/components/Home.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { Box, Heading, List, ListItem } from '@chakra-ui/react';

// Define posts with their names and dynamic paths
const posts = [
  { name: 'Erasing the Invisible Challenge', path: '/Erasing the Invisible' },
];

const Home: React.FC = () => (
  <Box p={4}>
    <Heading mb={4}>Blog Posts</Heading>
    <List spacing={3}>
      {posts.map((post) => (
        <ListItem key={post.path}>
          <Link to={post.path}>
            <Heading as="h3" size="md">
              {post.name}
            </Heading>
          </Link>
        </ListItem>
      ))}
    </List>
  </Box>
);

export default Home;
