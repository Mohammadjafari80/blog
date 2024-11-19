// src/components/Home.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import {
  Box,
  Heading,
  Text,
  List,
  ListItem,
  Grid,
  GridItem,
  Image,
} from '@chakra-ui/react';

// Define posts with their names, descriptions, images, and dynamic paths
const posts = [
  {
    name: 'Erasing the Invisible Challenge',
    path: '/Erasing the Invisible',
    description: 'My First Competition Experience.',
    image: 'erasing-the-invisible.jpg', // Example image path
  },
];

const Home: React.FC = () => (
  <Box p={4} minH="80vh">
    <Heading mb={6}>Blog Posts</Heading>
    <List spacing={5} dir="ltr" textAlign="left">
      {posts.map((post) => (
        <ListItem key={post.path}>
          <Link to={post.path}>
            <Grid
              templateColumns={['1fr', '100px 1fr']} // 1 column on small screens, 2 columns on larger screens
              gap={4}
              alignItems="center"
              p={4}
              borderWidth={1}
              borderRadius="md"
              boxShadow="md"
              _hover={{ boxShadow: 'lg', transform: 'scale(1.02)' }}
              transition="all 0.3s"
            >
              {/* Image on top for small screens, left for larger screens */}
              <GridItem colSpan={[1, 1]} rowSpan={[1, 2]}>
                <Image
                  src={post.image}
                  alt={post.name}
                  borderRadius="md"
                  objectFit="cover"
                  w={['100%', '100px']}
                  h={['auto', '100px']}
                  mx="auto" // Center the image for small screens
                />
              </GridItem>
              {/* Title and Description */}
              <GridItem>
                <Heading as="h2" size="lg" mb={2}>
                  {post.name}
                </Heading>
                <Text fontSize="sm" color="gray.600">
                  {post.description}
                </Text>
              </GridItem>
            </Grid>
          </Link>
        </ListItem>
      ))}
    </List>
  </Box>
);

export default Home;
