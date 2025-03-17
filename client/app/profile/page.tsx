"use client";

import React from "react";
import { motion } from "framer-motion";
import ElectionMini from "../components/Cards/ElectionMini";
import { useOpenElection } from "../components/Hooks/GetOpenElections";
import Loader from "../components/Helper/Loader";

const ElectionMiniSkeleton: React.FC = () => (
  <div className="bg-white p-6 rounded-lg shadow-md animate-pulse w-full">
    <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
    <div className="h-4 bg-gray-200 rounded w-1/2 mb-4"></div>
    <div className="h-4 bg-gray-200 rounded w-5/6 mb-4"></div>
    <div className="h-8 bg-gray-200 rounded w-1/4 mt-6"></div>
  </div>
);

const ProfilePage: React.FC = () => {
  const { elections = [], isLoading } = useOpenElection();

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-100 to-gray-200 flex justify-center py-10">
      <div className="w-full max-w-4xl p-6 bg-white rounded-3xl shadow-xl flex flex-col">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">Profile</h1>
        {isLoading ? (
          <div className="flex flex-col items-center justify-center space-y-4">
            <Loader />
            <p className="text-gray-500">Loading elections...</p>
          </div>
        ) : elections.length === 0 ? (
          <div className="text-center text-gray-500 py-8">No elections found</div>
        ) : (
          <motion.div
            className="space-y-6 overflow-y-auto pr-4"
            style={{ maxHeight: "70vh" }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {elections.map((election, index) => (
              <motion.div 
                key={election || index} 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <ElectionMini electionAddress={election} />
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default ProfilePage;
