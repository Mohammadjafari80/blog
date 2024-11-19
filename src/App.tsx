// src/App.tsx
import React from 'react';
import { Routes, Route, HashRouter } from 'react-router-dom';
import Home from './components/Home';
import Post from './components/Post';
import NavBar from './components/NavBar';
import Section from './components/Section';
import { Box, Center, Text } from '@chakra-ui/react';
import VisitorMap from './components/VisitorMap';
import './App.css';

const App: React.FC = () => {
  return (
    <HashRouter>
      <NavBar />

      {/* Main Content Section */}
      <Section variant="dark">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/:postName" element={<Post />} />
        </Routes>
      </Section>

      {/* Visitor Stats and Map Section */}
      <Section variant="dark">
        <Center mt={-3}>
          <a
            href="https://hits.seeyoufarm.com"
            aria-label="Visitor count for portfolio"
          >
            <img
              src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMohammadjafari80%2FMohammadjafari80.github.io&count_bg=%23D05A45&title_bg=%23373232&icon=&icon_color=%23E7E7E7&title=visits&edge_flat=false"
              alt="visitor count badge"
            />
          </a>
        </Center>
        <Center>
          <VisitorMap />
        </Center>
      </Section>

      {/* Footer Section */}
      <Box textAlign="center" mb={8}>
        <Text fontSize="sm">Designed by Mohammad Jafari. All rights reserved.</Text>
        <Text fontSize="sm">This design is original and created for personal use.</Text>
      </Box>
    </HashRouter>
  );
};

export default App;
