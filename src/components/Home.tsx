// src/components/Home.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import {
  Box,
  Heading,
  Text,
  List,
  ListItem,
  Flex,
  Image,
} from '@chakra-ui/react';

// Define posts with their names, descriptions, images, and dynamic paths
const posts = [
  {
    name: 'Erasing the Invisible Challenge',
    path: '/Erasing the Invisible',
    description: 'My First Competition Experience.',
    image: 'erasing-the-invisible.jpg', // Example image path
  }
];

const Home: React.FC = () => (
  <Box p={4} minH={'80vh'}>
    <Heading mb={6}>Blog Posts</Heading>
    <List spacing={5} dir="ltr" textAlign="left">
      {posts.map((post) => (
        <ListItem key={post.path}>
          <Link to={post.path}>
            <Flex
              direction={['column', 'row']}
              alignItems="center"
              p={4}
              borderWidth={1}
              borderRadius="md"
              boxShadow="md"
              _hover={{ boxShadow: 'lg', transform: 'scale(1.02)' }}
              transition="all 0.3s"
            >
              <Box flex="1" ml={4}>
                <Heading as="h2" size="lg" mb={2} >
                  {post.name}
                </Heading>
                <Text fontSize="sm" color="gray.600">
                  {post.description}
                </Text>
              </Box>
              <Image
                src={post.image}
                alt={post.name}
                boxSize="100px"
                borderRadius="md"
                objectFit="cover"
                ml={[0, 4]}
                mt={[4, 0]}
              />
            </Flex>
          </Link>
        </ListItem>
      ))}
    </List>
  </Box>
);

export default Home;
